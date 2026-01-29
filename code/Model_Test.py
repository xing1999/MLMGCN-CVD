# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn import metrics
from Modeling import NPZSeqDataset, SeqClassifier  # 根据实际情况修改路径

# 路径
FEAT_ROOT   = r"data\test\feats"
LABELS_CSV  = r"data\test\labels.csv"
SAVE_DIR    = Path("checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_identity(batch): return batch

# ==================
@torch.no_grad()

# ========= 评估函数 =========
@torch.no_grad()
def evaluate(model, loader, thr=0.5):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        for sid, s_np, nt_np, shp_np, (idx_np, wts_np), ntseq_np, y in batch:
            s = torch.from_numpy(s_np).to(DEVICE)
            nt = torch.from_numpy(nt_np).to(DEVICE)
            shp = torch.from_numpy(shp_np).to(DEVICE)
            ntseq = torch.from_numpy(ntseq_np).to(DEVICE)
            logit = model(s, nt, shp, idx_np, wts_np, ntseq)
            prob = torch.sigmoid(logit).item()
            y_true.append(int(y))
            y_prob.append(prob)

    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    f1  = metrics.f1_score(y_true, y_pred, zero_division=0)   # ★ 新增
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
    sn = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    return {"ACC": acc, "AUC": auc, "MCC": mcc, "F1": f1, "SN": sn, "SP": sp}

# ==================
def main():
    ds = NPZSeqDataset(
        labels_csv=LABELS_CSV,
        feat_root=FEAT_ROOT,
        struct_mode="shape",  # Optional 'ss' (Single-chain structural characteristics) or 'shape' (DNAshape)
        use_nt_seq=True
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_identity)


    _, s0, nt0, shp0, _, ntseq0, _ = ds[0]
    dim_s, dim_nt, dim_shape = s0.shape[1], nt0.shape[1], shp0.shape[1]

    results = []
    for fold in range(1, 11):
        ckpt = SAVE_DIR / f"DS_fold{fold}_best.pt"
        thr_path = SAVE_DIR / f"hreshold_DS_fold{fold}.txt"
        if not ckpt.exists():
            print(f"[Warn] {ckpt} not found, skip")
            continue

        print(f"Loading {ckpt}")
        model = SeqClassifier(
            nlayers=6, dim_s=dim_s, dim_nt=dim_nt, dim_shape=dim_shape,
            hidden_dim=768, dropout=0.25, lamda=1.5, alpha=0.7,
            variant=False, use_content_masked=True, num_queries=4
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        thr = 0.5
        if thr_path.exists():
            thr = float(open(thr_path).read().strip())

        metrics_dict = evaluate(model, loader, thr)
        metrics_dict["fold"] = fold
        metrics_dict["thr"] = thr
        results.append(metrics_dict)

    df = pd.DataFrame(results)
    print("\n===== Test Results =====")
    print(df)
    print("Mean ACC: {:.3f} ± {:.3f}".format(df["ACC"].mean(), df["ACC"].std()))
    print("Mean AUC: {:.3f} ± {:.3f}".format(df["AUC"].mean(), df["AUC"].std()))
    out_csv = SAVE_DIR / "test_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved summary to:", out_csv)

if __name__ == "__main__":
    main()


# import numpy as np
# import pandas as pd
# import torch
# from pathlib import Path
# from torch.utils.data import DataLoader
# from sklearn import metrics
#
# from Modeling import NPZSeqDataset, SeqClassifier  # 按你工程实际路径改
#
# # ================== ==================
# FEAT_ROOT = Path(r"data\test\feats")
# LABELS_CSV = Path(r"data\test\feats\labels.csv")

# DS_SAVE_DIR = Path(r"checkpoints\dsDNA-Model")
# SS_SAVE_DIR = Path(r"checkpoints\ssDNA-Model")
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def collate_identity(batch):
#     return batch
#
#
# def safe_load_state_dict(model, ckpt_path: Path):
#     obj = torch.load(ckpt_path, map_location=DEVICE)
#
#     if isinstance(obj, dict):
#         if "state_dict" in obj:
#             state = obj["state_dict"]
#         elif "model" in obj:
#             state = obj["model"]
#         else:
#             state = obj
#     else:
#         state = obj
#
#     if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
#         state = {k.replace("module.", "", 1): v for k, v in state.items()}
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing:
#         print(f"[Warn] missing keys ({len(missing)}): {missing[:5]} ...")
#     if unexpected:
#         print(f"[Warn] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
#
# def entropy_binary(p: float, eps: float = 1e-12) -> float:
#     p = float(np.clip(p, eps, 1.0 - eps))
#     return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
#
# def fix_knn_to_L(indices, weights, L_ref: int):
#     """ (indices, weights)       [L_ref, K]"""
#     idx = np.asarray(indices)
#     wts = np.asarray(weights)
#
#     while idx.ndim > 2 and 1 in idx.shape:
#         idx = np.squeeze(idx)
#     while wts.ndim > 2 and 1 in wts.shape:
#         wts = np.squeeze(wts)
#
#     # reshape
#     if idx.ndim == 1:
#         K = max(1, idx.size // max(1, L_ref))
#         idx = idx.reshape(-1, K)
#     if wts.ndim == 1:
#         K = max(1, wts.size // max(1, L_ref))
#         wts = wts.reshape(-1, K)
#
#     if idx.shape[0] != L_ref and idx.shape[1] == L_ref:
#         idx = idx.T
#         wts = wts.T
#
#     if idx.shape[0] > L_ref:
#         idx = idx[:L_ref]
#         wts = wts[:L_ref]
#     elif idx.shape[0] < L_ref:
#         need = L_ref - idx.shape[0]
#         idx = np.concatenate([idx, np.repeat(idx[-1:], need, axis=0)], axis=0)
#         wts = np.concatenate([wts, np.repeat(wts[-1:], need, axis=0)], axis=0)

#     K = min(idx.shape[1], wts.shape[1])
#     idx = idx[:, :K].astype(np.int64, copy=False)
#     wts = wts[:, :K].astype(np.float32, copy=False)
#
#     #  indices / weights
#     rows = np.arange(L_ref)[:, None]
#     idx = np.where((idx >= 0) & (idx < L_ref), idx, rows)
#     wts = np.where(np.isfinite(wts) & (wts > 0), wts, 0.0)

#     row_sum = wts.sum(axis=1, keepdims=True)
#     zero_row = (row_sum <= 1e-12).reshape(-1)
#     if np.any(zero_row):
#         idx[zero_row, :] = np.tile(np.arange(L_ref).reshape(-1, 1), (1, K))[zero_row]
#         wts[zero_row, :] = 0.0
#         wts[zero_row, 0] = 1.0
#         row_sum = wts.sum(axis=1, keepdims=True)
#
#     wts = wts / (row_sum + 1e-8)
#     return idx, wts
#
#
# def build_model(dim_s, dim_nt, dim_shape):
#     return SeqClassifier(
#         nlayers=6, dim_s=dim_s, dim_nt=dim_nt, dim_shape=dim_shape,
#         hidden_dim=768, dropout=0.25, lamda=1.5, alpha=0.7,
#         variant=False, use_content_masked=True, num_queries=4
#     ).to(DEVICE)
#
# @torch.no_grad()
# def evaluate_screen_review(
#     ds_model,
#     ss_model,
#     loader_ds,
#     loader_ss,
#     thr=0.5,
#     delta=0.05,
#     tau=None
# ):
#     ds_model.eval()
#     ss_model.eval()
#
#     y_true, y_prob = [], []
#     review_cnt, total_cnt = 0, 0
#
#     for batch_ds, batch_ss in zip(loader_ds, loader_ss):
#         # batch_size=1 -> list(len=1)
#         for sample_ds, sample_ss in zip(batch_ds, batch_ss):
#             sid_ds, s_ds, nt_ds, shp_ds, (idx_ds, wts_ds), ntseq_ds, y_ds = sample_ds
#             sid_ss, s_ss, nt_ss, shp_ss, (idx_ss, wts_ss), ntseq_ss, y_ss = sample_ss
#
#             if sid_ds != sid_ss:
#                 raise RuntimeError(f"Sample mismatch: {sid_ds} vs {sid_ss}")
#             if int(y_ds) != int(y_ss):
#                 raise RuntimeError(f"Label mismatch at {sid_ds}: {y_ds} vs {y_ss}")
#
#             # -------
#             L_common = min(
#                 s_ds.shape[0], nt_ds.shape[0], shp_ds.shape[0],
#                 s_ss.shape[0], nt_ss.shape[0], shp_ss.shape[0]
#             )
#
#             s_np     = s_ds[:L_common]
#             nt_np    = nt_ds[:L_common]
#             shpds_np = shp_ds[:L_common]
#             shpss_np = shp_ss[:L_common]
#
#             idx_np, wts_np = fix_knn_to_L(idx_ds, wts_ds, L_common)
#
#             s     = torch.from_numpy(s_np).to(DEVICE)
#             nt    = torch.from_numpy(nt_np).to(DEVICE)
#             shpds = torch.from_numpy(shpds_np).to(DEVICE)
#             shpss = torch.from_numpy(shpss_np).to(DEVICE)
#
#             # nt_seq  [Dn]，ds/ss
#             ntseq = torch.from_numpy(ntseq_ds).to(DEVICE)
#
#             # -------- dsDNA --------
#             logit_ds = ds_model(s, nt, shpds, idx_np, wts_np, ntseq)
#             p_ds = torch.sigmoid(logit_ds).item()
#
#             # -------- --------
#             uncertain = (abs(p_ds - 0.5) < float(delta))
#             if tau is not None:
#                 uncertain = uncertain or (entropy_binary(p_ds) > float(tau))
#
#             # -------- ssDNA --------
#             if uncertain:
#                 logit_ss = ss_model(s, nt, shpss, idx_np, wts_np, ntseq)
#                 p_final = torch.sigmoid(logit_ss).item()
#                 review_cnt += 1
#             else:
#                 p_final = p_ds
#
#             y_true.append(int(y_ds))
#             y_prob.append(float(p_final))
#             total_cnt += 1
#
#     y_true = np.array(y_true)
#     y_prob = np.array(y_prob)
#     y_pred = (y_prob >= thr).astype(int)
#
#     acc = metrics.accuracy_score(y_true, y_pred)
#     try:
#         auc = metrics.roc_auc_score(y_true, y_prob)
#     except Exception:
#         auc = float("nan")
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     f1  = metrics.f1_score(y_true, y_pred, zero_division=0)
#
#     cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
#     tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
#     sn = tp / (tp + fn + 1e-8)
#     sp = tn / (tn + fp + 1e-8)
#
#     return {
#         "ACC": acc, "AUC": auc, "MCC": mcc, "F1": f1, "SN": sn, "SP": sp,
#         "review_cnt": review_cnt,
#         "review_rate": review_cnt / (total_cnt + 1e-12),
#         "N": total_cnt
#     }
#
#
# def main():
#     ds_ds = NPZSeqDataset(labels_csv=str(LABELS_CSV), feat_root=str(FEAT_ROOT),
#                           struct_mode="shape", use_nt_seq=True)
#     ds_ss = NPZSeqDataset(labels_csv=str(LABELS_CSV), feat_root=str(FEAT_ROOT),
#                           struct_mode="ss", use_nt_seq=True)
#
#     loader_ds = DataLoader(ds_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_identity)
#     loader_ss = DataLoader(ds_ss, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_identity)
#
#
#     _, s0, nt0, shp0_ds, _, ntseq0, _ = ds_ds[0]
#     _, _,  _,  shp0_ss, _, _,      _ = ds_ss[0]
#     dim_s, dim_nt = s0.shape[1], nt0.shape[1]
#     dim_shape_ds, dim_shape_ss = shp0_ds.shape[1], shp0_ss.shape[1]
#
#     results = []
#
#     for fold in range(1, 11):
#         ckpt_ds = DS_SAVE_DIR / f"DS_fold{fold}_best.pt"
#         ckpt_ss = SS_SAVE_DIR / f"SS_fold{fold}_best.pt"
#
#         if (not ckpt_ds.exists()) or (not ckpt_ss.exists()):
#             print(f"[Warn] missing ckpt (fold={fold}): ds={ckpt_ds.exists()} ss={ckpt_ss.exists()} -> skip")
#             continue
#
#         model_ds = build_model(dim_s, dim_nt, dim_shape_ds)
#         model_ss = build_model(dim_s, dim_nt, dim_shape_ss)
#         safe_load_state_dict(model_ds, ckpt_ds)
#         safe_load_state_dict(model_ss, ckpt_ss)
#
#         thr = 0.5
#         thr_path = DS_SAVE_DIR / f"best_threshold_DS_fold{fold}.txt"
#         if thr_path.exists():
#             thr = float(thr_path.read_text(encoding="utf-8").strip())
#
#         delta = 0.05
#         delta_path = DS_SAVE_DIR / f"best_delta_fold{fold}.txt"
#         if delta_path.exists():
#             delta = float(delta_path.read_text(encoding="utf-8").strip())
#
#         tau = None
#         tau_path = DS_SAVE_DIR / f"best_tau_fold{fold}.txt"
#         if tau_path.exists():
#             tau = float(tau_path.read_text(encoding="utf-8").strip())
#
#         md = evaluate_screen_review(
#             model_ds, model_ss, loader_ds, loader_ss,
#             thr=thr, delta=delta, tau=tau
#         )
#         md.update({"fold": fold, "thr": thr, "delta": delta, "tau": (tau if tau is not None else np.nan)})
#         results.append(md)
#
#         print(f"[Fold {fold}] ACC={md['ACC']:.4f} AUC={md['AUC']:.4f} "
#               f"F1={md['F1']:.4f} MCC={md['MCC']:.4f} review={md['review_rate']:.3f}")
#
#     df = pd.DataFrame(results)
#     print("\n===== Test Results (Screen-Review) =====")
#     print(df)
#
#     out_csv = DS_SAVE_DIR / "test_summary_screen_review.csv"
#     df.to_csv(out_csv, index=False, encoding="utf-8")
#     print("Saved summary to:", out_csv)
#
#
# if __name__ == "__main__":
#     main()
