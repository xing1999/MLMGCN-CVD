import os
os.environ["DISABLE_TRITON"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import re, subprocess, tempfile, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


CSV_PATH     = r"data\test\test.csv"
OUT_ROOT     = Path(r"data\test\feats")
CHUNK_SIZE   = 200

# ---- DNABERT-S ----
USE_DNABERTS = r"D:\hf_cache\hub\models--zhihan1996--DNABERT-S\snapshots\00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
K_NEIGHBORS  = 20

# ---- DNAshape----
ENABLE_DNASHAPE = True
RSCRIPT_EXE     = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

# ---- Nucleotide Transformer----
ENABLE_NT                = True
NT_MODEL_ID              = "InstaDeepAI/nucleotide-transformer-500m-1000g"
NT_BATCH_SIZE            = 8
NT_TRIM_PADDING          = True
NT_TOKEN_DTYPE           = np.float16
NT_SEQ_DTYPE             = np.float16
NT_USE_COMPRESSED_NPZ    = True
NT_SUBSAMPLE_STRIDE      = 2
NT_ENABLE_RANDOM_PROJ    = True
NT_RP_DIM                = 256
NT_RANDOM_SEED           = 2020

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 2020
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# =========================
def save_dnaberts_tok(path_no_suffix: Path, emb_tok: np.ndarray):
    np.savez_compressed(path_no_suffix.with_suffix(".npz"),
                        emb=emb_tok.astype(np.float16))

def ensure_dirs(root: Path):
    (root/"dnaberts_emb").mkdir(parents=True, exist_ok=True)
    (root/"dnaberts_attn").mkdir(parents=True, exist_ok=True)
    (root/"dnashape").mkdir(parents=True, exist_ok=True)
    (root/"nt_emb_tok").mkdir(parents=True, exist_ok=True)
    (root/"nt_emb_seq").mkdir(parents=True, exist_ok=True)

def clean_for_dnaberts(seq: str) -> str:
    return re.sub(r"[^ACGT]", "N", seq.upper())

def extract_last_layer_attn_to_LxL(att_last) -> torch.Tensor:
    if isinstance(att_last, (list, tuple)):
        att_last = att_last[-1]
    t = att_last
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.dim() == 4:     # [B,H,L,L]
        A = t.mean(dim=1)       # [B,L,L]
        return A[0] if A.size(0) > 0 else A.squeeze(0)
    elif t.dim() == 3:   # [H,L,L]
        return t.mean(dim=0)
    elif t.dim() == 2:   # [L,L]
        return t
    else:
        raise ValueError(f"Unexpected attention dim: {t.dim()} shape={tuple(t.shape)}")

def attn_to_knn_indices_weights(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    L = A.shape[0]
    if L == 0:
        return np.zeros((0, k), dtype=np.int32), np.zeros((0, k), dtype=np.float16)

    A = A.copy()
    np.fill_diagonal(A, -np.inf)
    k_eff = min(k, max(L - 1, 0))
    if k_eff == 0:
        idx = np.zeros((L, k), dtype=np.int32)
        for i in range(L):
            idx[i, :] = i
        wts = np.zeros((L, k), dtype=np.float16)
        return idx, wts

    part_idx  = np.argpartition(-A, kth=k_eff-1, axis=1)[:, :k_eff]   # [L,k_eff]
    rows      = np.arange(L)[:, None]
    part_vals = A[rows, part_idx]
    order     = np.argsort(-part_vals, axis=1)
    top_idx   = np.take_along_axis(part_idx, order, axis=1).astype(np.int32)
    top_vals  = np.take_along_axis(part_vals, order, axis=1)

    top_vals  = np.where(np.isfinite(top_vals) & (top_vals > 0), top_vals, 0.0)
    row_sums  = top_vals.sum(axis=1, keepdims=True) + 1e-8
    norm_vals = (top_vals / row_sums).astype(np.float32)

    if k_eff < k:
        pad_n    = k - k_eff
        pad_idx  = np.repeat(np.arange(L)[:, None], pad_n, axis=1).astype(np.int32)
        pad_vals = np.zeros((L, pad_n), dtype=np.float32)
        indices  = np.concatenate([top_idx, pad_idx], axis=1)
        weights  = np.concatenate([norm_vals, pad_vals], axis=1)
    else:
        indices, weights = top_idx, norm_vals

    return indices.astype(np.int32), weights.astype(np.float16)

def save_knn_npz(indices: np.ndarray, weights: np.ndarray, path_no_suffix: Path):
    np.savez_compressed(
        path_no_suffix.with_suffix(".npz"),
        indices=indices.astype(np.int32),
        weights=weights.astype(np.float16)
    )

def load_knn_npz(path_npz: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path_npz)
    return z["indices"].astype(np.int32), z["weights"]

def load_dnaberts():
    tok = AutoTokenizer.from_pretrained(USE_DNABERTS, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        USE_DNABERTS, trust_remote_code=True,
        output_attentions=True, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    return tok, mdl

def dnashape_batch_predict_and_save(seqs, gid_start: int, out_root: Path):
    tmp_dir = out_root / "_dnashape_tmp_batch"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fasta = tmp_dir / "_tmp_input.fasta"
    with open(fasta, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s.strip().upper()}\n")

    rcode = r"""
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(DNAshapeR))
f <- args[1]; o <- args[2]
dir.create(o, showWarnings=FALSE, recursive=TRUE)
pred <- getShape(f)
props <- intersect(names(pred), c("HelT","MGW","ProT","Roll"))
for (p in props) {
  vals <- pred[[p]]
  con <- file(file.path(o, paste0(p, ".shape")), open="wt")
  for (i in seq_along(vals)) {
    line <- paste(vals[[i]], collapse=" ")
    writeLines(line, con=con, sep="\n")
  }
  close(con)
}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as rf:
        rf.write(rcode)
        rfile = rf.name
    try:
        subprocess.run([RSCRIPT_EXE, rfile, str(fasta), str(tmp_dir)], check=True)
    finally:
        try: os.remove(rfile)
        except: pass

    props = ["HelT", "MGW", "ProT", "Roll"]
    prop_lines = {}
    for p in props:
        fp = tmp_dir / f"{p}.shape"
        if fp.exists():
            with open(fp, "r") as f:
                prop_lines[p] = [ln.strip() for ln in f if ln.strip()]
        else:
            prop_lines[p] = []

    save_dir = out_root / "dnashape"
    for local_idx, s in enumerate(seqs):
        gid = gid_start + local_idx
        L = len(s)
        cols = []
        for p in props:
            lines = prop_lines.get(p, [])
            if local_idx >= len(lines):
                vec = np.zeros((L,), dtype=np.float32)
            else:
                vals = []
                for v in lines[local_idx].split():
                    try: vals.append(float(v))
                    except: vals.append(0.0)
                vec = np.asarray(vals, dtype=np.float32)
                if vec.shape[0] < L:
                    vec = np.concatenate([vec, np.zeros((L-vec.shape[0],), dtype=np.float32)])
                elif vec.shape[0] > L:
                    vec = vec[:L]
            cols.append(vec)
        M = np.stack(cols, axis=-1)  # [L,4]
        np.save(save_dir / f"{gid:06d}.npy", M.astype(np.float32))

    shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================
def _nt_build_random_proj_matrix(in_dim: int, out_dim: int, seed: int = 2020) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.standard_normal(size=(in_dim, out_dim)).astype(np.float32)
    R /= np.sqrt(out_dim).astype(np.float32)
    return R

def _nt_save_token_embeddings(path_base: Path, emb_tok: np.ndarray):
    if NT_USE_COMPRESSED_NPZ:
        np.savez_compressed(path_base.with_suffix(".npz"), tok=emb_tok)
    else:
        np.save(path_base.with_suffix(".npy"), emb_tok)

def _nt_save_seq_embedding(path_base: Path, emb_seq: np.ndarray):
    np.save(path_base.with_suffix(".npy"), emb_seq.astype(NT_SEQ_DTYPE))

def _nt_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

# =========================
def main():
    root = OUT_ROOT
    ensure_dirs(root)

    print("[Init] Loading DNABERT-S ...")
    s_tok, s_mdl = load_dnaberts()

    if ENABLE_NT:
        print(f"[Init] Loading NT model {NT_MODEL_ID} ...")
        nt_tokenizer = AutoTokenizer.from_pretrained(NT_MODEL_ID)
        nt_model     = AutoModelForMaskedLM.from_pretrained(NT_MODEL_ID).to(DEVICE).eval()
        nt_max_len   = nt_tokenizer.model_max_length
        with torch.no_grad():
            dummy = nt_tokenizer("ACGT", return_tensors="pt", padding="max_length", max_length=16)
            dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
            out   = nt_model(**dummy, output_hidden_states=True)
            NT_D_IN = out["hidden_states"][-1].shape[-1]
        if NT_ENABLE_RANDOM_PROJ:
            assert NT_RP_DIM <= NT_D_IN, f"NT_RP_DIM={NT_RP_DIM} 不能大于 NT_D_IN={NT_D_IN}"
            NT_R = _nt_build_random_proj_matrix(NT_D_IN, NT_RP_DIM, NT_RANDOM_SEED)
            NT_D_OUT = NT_RP_DIM
            print(f"[Info][NT] Random projection: D_in={NT_D_IN} -> D_out={NT_D_OUT}")
        else:
            NT_R = None
            NT_D_OUT = NT_D_IN
            print(f"[Info][NT] No projection: D_out = {NT_D_OUT}")
    else:
        nt_tokenizer = None
        nt_model     = None
        nt_max_len   = None
        NT_R         = None
        NT_D_OUT     = None

    print("[Run] Extracting features by chunks ...")
    gid_base = 0

    with torch.no_grad():
        for df_chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
            if "sequence" not in df_chunk.columns:
                raise KeyError(f"CSV 缺少 'sequence' 列：{CSV_PATH}")
            seqs_raw = df_chunk["sequence"].astype(str).tolist()
            N = len(seqs_raw)

            # ========= DNABERT-S =========
            for local_i, seq_raw in enumerate(tqdm(seqs_raw, desc="DNABERT-S", leave=False)):
                gid = gid_base + local_i
                out_emb_base = root / "dnaberts_emb" / f"{gid:06d}"  # 不带后缀，保存成 .npz
                out_att_base = root / "dnaberts_attn" / f"{gid:06d}"  # .npz

                if out_emb_base.with_suffix(".npz").exists() and out_att_base.with_suffix(".npz").exists():
                    continue

                seq_s = clean_for_dnaberts(seq_raw)
                inp_s = s_tok(seq_s, return_tensors="pt")
                inp_s = {k: v.to(DEVICE) for k, v in inp_s.items()}
                out_s = s_mdl(**inp_s)

                if isinstance(out_s, tuple):
                    last_hidden = out_s[0]  # [1,L,D]
                    all_attn = out_s[-1]
                else:
                    last_hidden = out_s.last_hidden_state  # [1,L,D]
                    all_attn = out_s.attentions

                rep_s = last_hidden.squeeze(0)  # [L,D]
                att_last = all_attn[-1] if isinstance(all_attn, (list, tuple)) else all_attn
                A_LxL = extract_last_layer_attn_to_LxL(att_last)  # [L,L]

                np.savez_compressed(out_emb_base.with_suffix(".npz"),
                                    emb=rep_s.cpu().numpy().astype(np.float16))

                A_np = A_LxL.detach().cpu().numpy().astype(np.float32)
                knn_idx, knn_wts = attn_to_knn_indices_weights(A_np, K_NEIGHBORS)
                save_knn_npz(knn_idx, knn_wts, out_att_base)

            if ENABLE_NT:
                for start in tqdm(range(0, N, NT_BATCH_SIZE), desc="NT", leave=False):
                    batch = seqs_raw[start:start+NT_BATCH_SIZE]
                    tokens = nt_tokenizer.batch_encode_plus(
                        batch, return_tensors="pt", padding="max_length", max_length=nt_max_len
                    )
                    input_ids = tokens["input_ids"].to(DEVICE)
                    attention_mask = (input_ids != nt_tokenizer.pad_token_id).to(DEVICE)

                    outputs = nt_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    hidden = outputs["hidden_states"][-1]  # [B,L,D_in]

                    if NT_SUBSAMPLE_STRIDE > 1:
                        hidden = hidden[:, ::NT_SUBSAMPLE_STRIDE, :]
                        attention_mask = attention_mask[:, ::NT_SUBSAMPLE_STRIDE]

                    hidden_np = _nt_to_numpy(hidden)  # float32
                    mask_np   = attention_mask.unsqueeze(-1).cpu().numpy().astype(np.float32)  # [B, L', 1]

                    if NT_ENABLE_RANDOM_PROJ:
                        hidden_np = hidden_np @ NT_R  # (B,L,D_in)@(D_in,D_out)->(B,L,D_out)
                    hidden_np = hidden_np.astype(NT_TOKEN_DTYPE)
                    seq_emb = (hidden_np.astype(np.float32) * mask_np).sum(axis=1) / (mask_np.sum(axis=1) + 1e-8)
                    seq_emb = seq_emb.astype(NT_SEQ_DTYPE)

                    B_cur, Lp, _ = hidden_np.shape
                    for b in range(B_cur):
                        gid = gid_base + (start + b)
                        sid = f"{gid:06d}"

                        if NT_TRIM_PADDING:
                            l_valid = int(mask_np[b].sum().item())
                            l_valid = max(l_valid, 1)
                        else:
                            l_valid = Lp
                        tok_feat = hidden_np[b, :l_valid, :]  # [L_valid, D_out]

                        _nt_save_token_embeddings(root/"nt_emb_tok"/sid, tok_feat)
                        _nt_save_seq_embedding(root/"nt_emb_seq"/sid, seq_emb[b])

            # ========= DNAshape=========
            if ENABLE_DNASHAPE:
                dnashape_batch_predict_and_save(seqs_raw, gid_base, root)

            gid_base += N

    print("\n[Done] All features are saved under:", root)
if __name__ == "__main__":
    main()
