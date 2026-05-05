import os
os.environ["PATH"] += r";C:\Program Files (x86)\ViennaRNA Package"
import sys
import csv
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# =========I/O =========
def which(cmd: str) -> str:
    p = shutil.which(cmd)
    return p if p else ""

def have_rnafold() -> bool:
    return bool(which("RNAfold"))

def have_rnaplfold() -> bool:
    return bool(which("RNAplfold"))

def read_fasta(fp: str) -> List[Tuple[str, str]]:
    recs, sid, seq = [], None, []
    with open(fp, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith('>'):
                if sid is not None:
                    s = ''.join(seq).upper().replace(' ', '')
                    recs.append((sid, s))
                sid = ln[1:].split()[0]
                seq = []
            else:
                seq.append(ln)
        if sid is not None:
            s = ''.join(seq).upper().replace(' ', '')
            recs.append((sid, s))
    # DNA T->U
    recs = [(rid, rseq.replace('T', 'U')) for rid, rseq in recs]
    recs.sort(key=lambda x: x[0])
    return recs

def read_csv_sequences(fp: str,
                       seq_col: str = "sequence",
                       label_col: Optional[str] = "label") -> List[Tuple[str, str, Optional[int]]]:

    try:
        import pandas as pd
    except Exception:
        sys.exit(1)
    df = pd.read_csv(fp)
    assert seq_col in df.columns, f"CSV缺少列: {seq_col}"
    ids, seqs, labels = [], [], []
    for i, row in df.iterrows():
        ids.append(f"{i+1:06d}")
        seqs.append(str(row[seq_col]).strip().upper())
        labels.append(int(row[label_col]) if (label_col in df.columns) else None)
    return list(zip(ids, seqs, labels))

def csv_to_fasta(records: List[Tuple[str, str, Optional[int]]], fasta_out: str):
    with open(fasta_out, "w", encoding="utf-8") as f:
        for rid, dna, lab in records:
            lab_tag = f"_label{lab}" if lab is not None else ""
            f.write(f">{rid}{lab_tag}\n{dna}\n")

# ========= =========
def sliding_windows(seq: str, win: int, step: int) -> List[Tuple[int, int, str]]:
    L = len(seq)
    if win <= 0 or win >= L:
        return [(0, L, seq)]
    out, st = [], 0
    step = max(1, step)
    while st < L:
        ed = min(st + win, L)
        out.append((st, ed, seq[st:ed]))
        if ed >= L:
            break
        st += step
    return out

# ==================
def parse_rnafold_stdout(stdout: str) -> Tuple[str, float]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "", 0.0
    sline = lines[1]
    try:
        left, right = sline.rfind('('), sline.rfind(')')
        energy = 0.0
        if left != -1 and right != -1 and right > left:
            energy = float(sline[left+1:right].strip())
        dot = sline.split(' ')[0].strip()
        return dot, energy
    except Exception:
        return "", 0.0

def naive_unpaired_struct(seq: str) -> Tuple[str, float]:
    return "." * len(seq), 0.0

def run_rnafold_on_seq(seq_u: str) -> Tuple[str, float]:
    if not have_rnafold():
        return naive_unpaired_struct(seq_u)
    try:
        p = subprocess.run(["RNAfold", "--noPS", "--MEA"],
                           input=(seq_u + "\n").encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        dot, energy = parse_rnafold_stdout(p.stdout.decode("utf-8", errors="ignore"))
        if (not dot) or (len(dot) != len(seq_u)):
            return naive_unpaired_struct(seq_u)
        return dot, energy
    except Exception:
        return naive_unpaired_struct(seq_u)

def dot_to_paired_flags(dot: str) -> np.ndarray:
    if not dot:
        return np.zeros(0, dtype=np.float32)
    return np.array([0.0 if c == '.' else 1.0 for c in dot], dtype=np.float32)

def run_length_norm(flags: np.ndarray) -> np.ndarray:
    n = len(flags)
    out = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        j = i
        while j < n and flags[j] == flags[i]:
            j += 1
        out[i:j] = (j - i)
        i = j
    out /= max(1, n)
    return out

def norm_energy_per_base(energy: float, L: int) -> np.ndarray:
    if L <= 0:
        return np.zeros(0, dtype=np.float32)
    per_base = float(energy) / float(L)
    ref = -5.0
    val = per_base / abs(ref) if ref != 0 else 0.0
    val = max(-1.0, min(0.0, val))
    return np.full(L, val, dtype=np.float32)

def rnaplfold_unpaired_probs(seq_u: str, W: int = 200, Lspan: int = 120) -> np.ndarray:
    if not have_rnaplfold():
        return np.full(len(seq_u), np.nan, dtype=np.float32)
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            subprocess.run(["RNAplfold", "-W", str(W), "-L", str(Lspan), "-u", "1"],
                           input=(seq_u + "\n").encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            lunp = None
            for fn in os.listdir(tmp):
                if fn.endswith("_lunp"):
                    lunp = os.path.join(tmp, fn); break
            probs = np.full(len(seq_u), np.nan, dtype=np.float32)
            if lunp:
                with open(lunp, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if (not line) or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            pos = int(parts[0]) - 1
                            val = float(parts[1])
                            if 0 <= pos < len(seq_u):
                                probs[pos] = val
            if np.isnan(probs).any():
                idx = np.where(~np.isnan(probs))[0]
                if idx.size > 0:
                    for i in range(len(probs)):
                        if np.isnan(probs[i]):
                            j = idx[np.argmin(np.abs(idx - i))]
                            probs[i] = probs[j]
                else:
                    probs[:] = 0.5
            return probs.astype(np.float32)
        finally:
            os.chdir(cwd)

def stack_baselevel_feats(seq_u: str, use_plfold: bool) -> Tuple[np.ndarray, list, float]:
    dot, energy = run_rnafold_on_seq(seq_u)
    if not dot:
        dot, energy = "." * len(seq_u), 0.0
    paired = dot_to_paired_flags(dot)
    unpaired = 1.0 - paired
    region = run_length_norm(paired)
    mfe_norm = norm_energy_per_base(energy, len(seq_u))

    feats = [paired, unpaired, region, mfe_norm]
    keys = ["paired_flag", "unpaired_flag", "region_len_norm", "mfe_energy_norm"]

    if use_plfold:
        p_unp = rnaplfold_unpaired_probs(seq_u)
        if np.isnan(p_unp).any():
            proxy = 0.4 + 0.2 * unpaired
            p_unp = np.where(np.isnan(p_unp), proxy, p_unp).astype(np.float32)
        feats += [p_unp, 1.0 - p_unp]
        keys += ["p_unpaired_pf", "p_paired_pf"]

    X = np.stack(feats, axis=1).astype(np.float32)  # [L, C]
    return X, keys, float(energy)

def pool_to_kmers(X_base: np.ndarray, k: int) -> np.ndarray:
    L, C = X_base.shape
    if k <= 1:
        return X_base.copy()
    Lt = max(0, L - k + 1)
    if Lt == 0:
        return np.zeros((0, C), dtype=np.float32)
    out = np.zeros((Lt, C), dtype=np.float32)
    cs = np.cumsum(X_base, axis=0)
    for i in range(Lt):
        j = i + k - 1
        s = cs[j] if i == 0 else (cs[j] - cs[i-1])
        out[i] = s / float(k)
    return out

# ==================
def process_fasta(fasta_path: str,
                  outdir: str,
                  kmer: int = 6,
                  window: int = 0,
                  step: int = 0,
                  use_plfold: bool = False,
                  zpad: int = 6,
                  index_from: int = 1,
                  label_lookup: Optional[Dict[str, int]] = None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    map_path = outdir / "_index_map.csv"
    recs = read_fasta(fasta_path)  # (id, seqU)
    print(f"[Info] FASTA records = {len(recs)} ; RNAfold={have_rnafold()} ; RNAplfold={have_rnaplfold()}")

    with open(map_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["file","index","orig_id","label","start","end","seq_len","k","energy_mfe","keys"])
        counter, total = index_from, 0

        for rid, seq_u in recs:
            # label（>id_labelX）    label_lookup
            lab = None
            if "_label" in rid:
                try:
                    lab = int(rid.rsplit("_label", 1)[1])
                except Exception:
                    lab = None
            if lab is None and label_lookup is not None and rid in label_lookup:
                lab = int(label_lookup[rid])

            for (st, ed, subseq_u) in sliding_windows(seq_u, window, step):
                X_base, keys, energy = stack_baselevel_feats(subseq_u, use_plfold)
                X_tok = pool_to_kmers(X_base, kmer)
                meta = {
                    "id": rid, "start": int(st), "end": int(ed),
                    "seq_len": int(len(subseq_u)), "k": int(kmer),
                    "energy_mfe": float(energy), "feature_keys": keys
                }
                fname = f"{str(counter).zfill(zpad)}.npz"
                np.savez_compressed(outdir / fname, feats=X_tok, meta=json.dumps(meta))
                w.writerow([fname, counter, rid, lab, st, ed, len(subseq_u), kmer, energy, "|".join(keys)])
                counter += 1; total += 1

    print(f"[Done] wrote {total} files to {str(outdir.resolve())}")
    print(f"[Map ] index saved to {str(map_path.resolve())}")

def process_csv(csv_path: str,
                outdir: str,
                kmer: int = 6,
                window: int = 0,
                step: int = 0,
                use_plfold: bool = False,
                zpad: int = 6,
                index_from: int = 1,
                seq_col: str = "sequence",
                label_col: str = "label"):

    from tempfile import TemporaryDirectory
    records = read_csv_sequences(csv_path, seq_col=seq_col, label_col=label_col)  # (id6, DNA, label)
    label_lookup = {rid: lab for rid, _, lab in records}
    with TemporaryDirectory() as tmp:
        tmp_fa = Path(tmp) / "tmp.fa"
        csv_to_fasta(records, str(tmp_fa))
        process_fasta(str(tmp_fa), outdir, kmer, window, step, use_plfold, zpad, index_from, label_lookup)

# ========= CLI =========
def build_argparser():
    ap = argparse.ArgumentParser(description="Extract ssDNA (RNA-like) structure features -> 000001.npz order.")
    ap.add_argument("--fasta", type=str, default="", help="FASTA 输入（DNA），内部 T->U 折叠")
    ap.add_argument("--csv", type=str, default="", help="CSV 输入（需包含 sequence[,label] 列）")
    ap.add_argument("--outdir", type=str, required=False, default="data/train/feats", help="输出目录")
    ap.add_argument("--kmer", type=int, default=6)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--use-plfold", action="store_true")
    ap.add_argument("--zpad", type=int, default=6)
    ap.add_argument("--index-from", type=int, default=1)
    ap.add_argument("--seq-col", type=str, default="sequence")
    ap.add_argument("--label-col", type=str, default="label")
    return ap

def main_cli(args=None):
    ap = build_argparser()
    if args is None:
        args = ap.parse_args()
    else:
        args = ap.parse_args(args)

    if args.csv and Path(args.csv).exists():
        process_csv(args.csv, args.outdir, args.kmer, args.window, args.step,
                    args.use_plfold, args.zpad, args.index_from, args.seq_col, args.label_col)
    elif args.fasta and Path(args.fasta).exists():
        process_fasta(args.fasta, args.outdir, args.kmer, args.window, args.step,
                      args.use_plfold, args.zpad, args.index_from, label_lookup=None)
    else:
        print("[Error] 请提供 --csv 或 --fasta，并确保路径存在。", file=sys.stderr)
        sys.exit(1)

DEFAULT_INPUT_MODE = "csv"   # "csv" 或 "fasta"
DEFAULT_CSV_PATH   = "data/data.csv"      #  sequence,label
DEFAULT_FASTA_PATH = "data/sequences.fa"  #
DEFAULT_OUTDIR     = "data/train/feats/ss_struct"
DEFAULT_KMER       = 6
DEFAULT_WINDOW     = 0
DEFAULT_STEP       = 0
DEFAULT_USE_PLFOLD = False  # RNAplfold True
DEFAULT_ZPAD       = 6
DEFAULT_INDEX_FROM = 0
DEFAULT_SEQ_COL    = "sequence"
DEFAULT_LABEL_COL  = "label"

if __name__ == "__main__":
    if len(sys.argv) == 1:
        if DEFAULT_INPUT_MODE.lower() == "csv":
            process_csv(csv_path=DEFAULT_CSV_PATH,
                        outdir=DEFAULT_OUTDIR,
                        kmer=DEFAULT_KMER,
                        window=DEFAULT_WINDOW,
                        step=DEFAULT_STEP,
                        use_plfold=DEFAULT_USE_PLFOLD,
                        zpad=DEFAULT_ZPAD,
                        index_from=DEFAULT_INDEX_FROM,
                        seq_col=DEFAULT_SEQ_COL,
                        label_col=DEFAULT_LABEL_COL)
        else:
            process_fasta(fasta_path=DEFAULT_FASTA_PATH,
                          outdir=DEFAULT_OUTDIR,
                          kmer=DEFAULT_KMER,
                          window=DEFAULT_WINDOW,
                          step=DEFAULT_STEP,
                          use_plfold=DEFAULT_USE_PLFOLD,
                          zpad=DEFAULT_ZPAD,
                          index_from=DEFAULT_INDEX_FROM,
                          label_lookup=None)
    else:

        main_cli()
