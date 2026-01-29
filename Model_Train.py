# -*- coding: utf-8 -*-
import os, random, inspect
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


from Modeling import NPZSeqDataset, SeqClassifier

# ==============
FEAT_ROOT   = r"data\train\feats"
LABELS_CSV  = r"data\train\labels.csv"
SAVE_DIR    = Path(r"checkpoints")

N_SPLITS          = 10
EPOCHS            = 30
INIT_LR           = 3e-4
WEIGHT_DECAY      = 1e-5
DROP_OUT          = 0.25
LAYER             = 6
HIDDEN_DIM        = 768
LAMBDA            = 1.5
ALPHA             = 0.7
VARIANT           = False
USE_CONTENT_MASKED = True
SELF_LOOP_EPS     = 0.10

# batch
DL_BATCH_SIZE     = 16
ACCUM_STEPS       = 2
NUM_WORKERS       = 0
SEED              = 2020
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SELECT_METRIC     = 'ACC'

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# --------------------------------------------------
def _calc_all_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1  = metrics.f1_score(y_true, y_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))==2 else float("nan")
    try:
        auc = metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    except Exception:
        auc = float("nan")
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size==4 else (0,0,0,0))
    sn = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    return {"ACC": acc, "AUC": auc, "MCC": mcc, "SN": sn, "SP": sp, "F1": f1}

def _best_threshold_by_metric(y_true, y_prob, metric='ACC'):
    cand = np.linspace(0.05, 0.95, 37)
    best_val, best_thr, best_m = -1.0, 0.5, None
    for t in cand:
        m = _calc_all_metrics(y_true, y_prob, t)
        val = m.get(metric, -1.0)
        if np.isnan(val):
            continue
        if val > best_val:
            best_val, best_thr, best_m = val, float(t), m
    best_m["THR"] = best_thr
    return best_m

# ------------------------- Padding 工具 -------------------------
def pad_2d_to_len(t: torch.Tensor, target_len: int, pad_value: float = 0.0):
    """ t: [L, C] -> 右侧 pad 到 [target_len, C] （保持在 CPU） """
    L, C = t.shape
    if L == target_len:
        return t
    pad_amount = target_len - L
    t_ch_last = t.transpose(0, 1)                        # [C, L]
    t_padded  = F.pad(t_ch_last, (0, pad_amount), value=pad_value)  # [C, target_len]
    return t_padded.transpose(0, 1)                      # [target_len, C]

def pad_1d_to_len(t: torch.Tensor, target_len: int, pad_value: int = 0):
    """ t: [L] -> 右侧 pad 到 [target_len] （保持在 CPU） """
    L = t.shape[0]
    if L == target_len:
        return t
    pad_amount = target_len - L
    return F.pad(t, (0, pad_amount), value=pad_value)

# ------------------------- collate：变长 -> 批次对齐（仅 CPU） -------------------------
def collate_and_pad_cpu(batch):
    """
  (sid, s_np, nt_np, shp_np, (idx_np, wts_np), nt_seq_np, y)
    """
    sid_list = []
    s_list, nt_list, shp_list, ntseq_list = [], [], [], []
    idx_list, wts_list, y_list = [], [], []

    # numpy -> CPU tensor
    for sid, s_np, nt_np, shp_np, (idx_np, wts_np), nt_seq_np, y in batch:
        sid_list.append(sid)
        s_list.append(torch.from_numpy(s_np))           # [L_s, C_s]
        nt_list.append(torch.from_numpy(nt_np))         # [L_n, C_n]
        shp_list.append(torch.from_numpy(shp_np))       # [L_h, C_h]
        ntseq_list.append(torch.from_numpy(nt_seq_np))  #  [L_q, D]
        idx_list.append(torch.from_numpy(idx_np))       #
        wts_list.append(torch.from_numpy(wts_np))
        y_list.append(float(y))

    len_s   = [t.shape[0] for t in s_list]
    len_nt  = [t.shape[0] for t in nt_list]
    len_shp = [t.shape[0] for t in shp_list]
    len_seq = [t.shape[0] for t in ntseq_list]

    max_s, max_nt, max_shp, max_seq = max(len_s), max(len_nt), max(len_shp), max(len_seq)

    # pad + mask
    s_pad, nt_pad, shp_pad, ntseq_pad = [], [], [], []
    s_mask, nt_mask, shp_mask, ntseq_mask = [], [], [], []

    for s_t, n_t, h_t, q_t, Ls, Ln, Lh, Lq in zip(
        s_list, nt_list, shp_list, ntseq_list, len_s, len_nt, len_shp, len_seq
    ):
        s_pad.append(  pad_2d_to_len(s_t, max_s,  pad_value=0.0) )
        nt_pad.append( pad_2d_to_len(n_t, max_nt, pad_value=0.0) )
        shp_pad.append( pad_2d_to_len(h_t, max_shp, pad_value=0.0) )

        if q_t.ndim == 1:
            ntseq_pad.append( pad_1d_to_len(q_t, max_seq, pad_value=0) )
        else:
            q2 = pad_2d_to_len(q_t, max_seq, pad_value=0.0)
            ntseq_pad.append(q2)

        s_mask.append(    torch.cat([torch.ones(Ls), torch.zeros(max_s  - Ls)]) )
        nt_mask.append(   torch.cat([torch.ones(Ln), torch.zeros(max_nt - Ln)]) )
        shp_mask.append(  torch.cat([torch.ones(Lh), torch.zeros(max_shp- Lh)]) )
        ntseq_mask.append(torch.cat([torch.ones(Lq), torch.zeros(max_seq- Lq)]) )

    s    = torch.stack(s_pad,    dim=0)   # [B, max_s,   C_s]
    nt   = torch.stack(nt_pad,   dim=0)   # [B, max_nt,  C_n]
    shp  = torch.stack(shp_pad,  dim=0)   # [B, max_shp, C_h]
    if ntseq_pad[0].ndim == 1:
        ntseq = torch.stack(ntseq_pad, dim=0)           # [B, max_seq]
    else:
        ntseq = torch.stack(ntseq_pad, dim=0)           # [B, max_seq, D]

    mask = {
        "s":    torch.stack(s_mask,    dim=0),  # [B, max_s]
        "nt":   torch.stack(nt_mask,   dim=0),  # [B, max_nt]
        "shp":  torch.stack(shp_mask,  dim=0),  # [B, max_shp]
        "seq":  torch.stack(ntseq_mask,dim=0),  # [B, max_seq]
    }

    idx = idx_list
    wts = wts_list
    y   = torch.tensor(y_list, dtype=torch.float32)     # [B]

    return sid_list, s, nt, shp, (idx, wts), ntseq, y, mask

def make_loaders(ds, tr_idx, va_idx):
    tr_set, va_set = Subset(ds, tr_idx), Subset(ds, va_idx)
    pin = (DEVICE.type == "cuda")
    tr_loader = DataLoader(tr_set, batch_size=DL_BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, collate_fn=collate_and_pad_cpu, pin_memory=pin)
    va_loader = DataLoader(va_set, batch_size=DL_BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, collate_fn=collate_and_pad_cpu, pin_memory=pin)
    return tr_loader, va_loader

# ------------------------------------------------
def call_model(model, s, nt, shp, idx, wts, ntseq, mask=None):
    sig = inspect.signature(model.forward)
    if "mask" in sig.parameters:
        return model(s, nt, shp, idx, wts, ntseq, mask)
    else:
        return model(s, nt, shp, idx, wts, ntseq)

# ------------------------------------------------
def _to_device_batch(s, nt, shp, idx, wts, ntseq, y, mask):
    s    = s.to(DEVICE, non_blocking=True)
    nt   = nt.to(DEVICE, non_blocking=True)
    shp  = shp.to(DEVICE, non_blocking=True)
    if ntseq.ndim == 2:
        ntseq = ntseq.to(DEVICE, non_blocking=True)                 # [B, L]
    else:
        ntseq = ntseq.to(DEVICE, non_blocking=True)                 # [B, L, D]
    y    = y.to(DEVICE, non_blocking=True)
    idx  = [t.to(DEVICE, non_blocking=True) for t in idx]
    wts  = [t.to(DEVICE, non_blocking=True) for t in wts]
    mask = {k: v.to(DEVICE, non_blocking=True) for k, v in mask.items()}
    return s, nt, shp, idx, wts, ntseq, y, mask

def train_one_epoch(model, loader, optim, criterion):
    model.train()
    optim.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0
    for _, s, nt, shp, (idx, wts), ntseq, y, mask in loader:
        s, nt, shp, idx, wts, ntseq, y, mask = _to_device_batch(s, nt, shp, idx, wts, ntseq, y, mask)
        logit = call_model(model, s, nt, shp, idx, wts, ntseq, mask).view(-1)  # [B]
        loss  = criterion(logit, y)
        (loss / ACCUM_STEPS).backward()
        total_loss += loss.item()
        steps += 1
        if steps % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); optim.zero_grad(set_to_none=True)
    if steps % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step(); optim.zero_grad(set_to_none=True)
    return total_loss / max(1, steps)

@torch.no_grad()
def evaluate_find_thr(model, loader, criterion):
    model.eval()
    losses, y_true, y_prob = [], [], []
    for _, s, nt, shp, (idx, wts), ntseq, y, mask in loader:
        s, nt, shp, idx, wts, ntseq, y, mask = _to_device_batch(s, nt, shp, idx, wts, ntseq, y, mask)
        logit = call_model(model, s, nt, shp, idx, wts, ntseq, mask).view(-1)
        loss  = criterion(logit, y)
        losses.append(loss.item())
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logit).detach().cpu().numpy())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    best_m = _best_threshold_by_metric(y_true, y_prob, metric=SELECT_METRIC)
    return float(np.mean(losses)), best_m

# --------------------------------------------------
def build_model(sample_dim):
    dim_s, dim_nt, dim_shape = sample_dim
    model = SeqClassifier(
        nlayers=LAYER, dim_s=dim_s, dim_nt=dim_nt, dim_shape=dim_shape,
        hidden_dim=HIDDEN_DIM, dropout=DROP_OUT,
        lamda=LAMBDA, alpha=ALPHA, variant=VARIANT,
        use_content_masked=USE_CONTENT_MASKED, num_queries=4
    ).to(DEVICE)
    return model

# --------------------------------------------------
def main():
    # ds = NPZSeqDataset(LABELS_CSV, FEAT_ROOT)

    ds = NPZSeqDataset(
        labels_csv=LABELS_CSV,
        feat_root=FEAT_ROOT,
        struct_mode="shape",  # Optional 'ss' (Single-chain structural characteristics) or 'shape' (DNAshape)
        use_nt_seq=True
    )

    y_all = np.array(ds.labels, dtype=int)
    print("Label balance:", np.bincount(y_all))

    #  s/nt/shp    [L, C]）
    _, s0, nt0, shp0, _, ntseq0, _ = ds[0]
    dim_s   = s0.shape[-1]
    dim_nt  = nt0.shape[-1]
    dim_shp = shp0.shape[-1]
    print(f"[Info] dims: Ds={dim_s}, Dn={dim_nt}, Shape={dim_shp}, "
          f"Dn_seq_dim={ntseq0.shape[1] if hasattr(ntseq0, 'ndim') and ntseq0.ndim==2 else 1}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_logs, history_all, best_paths = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
        tr_loader, va_loader = make_loaders(ds, tr_idx, va_idx)

        pos = int((y_all[tr_idx] == 1).sum()); neg = int((y_all[tr_idx] == 0).sum())
        pos_weight = torch.tensor([neg / max(1, pos)], device=DEVICE, dtype=torch.float32)
        print(f"pos_weight = {pos_weight.item():.3f}  (neg={neg}, pos={pos})")

        model = build_model((dim_s, dim_nt, dim_shp))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=4)

        best_sel, best_auc, best_thr = -1.0, -1.0, 0.5
        best_path = SAVE_DIR / f"DS_fold{fold}_best.pt"
        thr_path  = SAVE_DIR / f"best_threshold_DS_fold{fold}.txt"
        no_improve, fold_history = 0, []

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, tr_loader, optim, criterion)
            va_loss, va_metrics = evaluate_find_thr(model, va_loader, criterion)

            sel_val = va_metrics[SELECT_METRIC]
            scheduler.step(sel_val)
            cur_lr = optim.param_groups[0]['lr']

            print(f"[F{fold:02d} E{epoch:03d}] train={tr_loss:.4f}  val={va_loss:.4f} | "
                  f"ACC={va_metrics['ACC']:.3f} AUC={va_metrics['AUC']:.3f} "
                  f"MCC={va_metrics['MCC']:.3f} F1={va_metrics['F1']:.3f} "
                  f"SN={va_metrics['SN']:.3f} SP={va_metrics['SP']:.3f} "
                  f"THR={va_metrics['THR']:.3f} | LR={cur_lr:.6f}")

            row = {
                "fold": fold, "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                "ACC": va_metrics["ACC"], "AUC": va_metrics["AUC"], "MCC": va_metrics["MCC"],
                "F1": va_metrics["F1"], "SN": va_metrics["SN"], "SP": va_metrics["SP"],
                "THR": va_metrics["THR"], "LR": cur_lr, "SELECT": sel_val
            }
            fold_history.append(row)
            history_all.append(row.copy())

            cur_auc = va_metrics["AUC"]
            improved = (sel_val > best_sel) or (abs(sel_val - best_sel) < 1e-8 and cur_auc > best_auc)
            if not np.isnan(sel_val) and improved:
                best_sel, best_auc, best_thr = sel_val, cur_auc, va_metrics["THR"]
                torch.save(model.state_dict(), best_path)
                with open(thr_path, "w", encoding="utf-8") as f:
                    f.write(f"{best_thr:.6f}\n")
                print(f"  -> saved best fold model to {best_path} "
                      f"({SELECT_METRIC}={best_sel:.3f}, AUC={best_auc:.3f}, THR={best_thr:.3f})")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 8:
                    print("  -> early stop this fold")
                    break

        df_fold = pd.DataFrame(fold_history)
        df_fold.to_csv(SAVE_DIR / f"history_fold{fold}.csv", index=False, encoding="utf-8")
        print("Saved history of fold", fold, "to:", SAVE_DIR / f"history_fold{fold}.csv")

        fold_logs.append({"fold": fold, SELECT_METRIC: best_sel, "AUC": best_auc, "THR": best_thr})
        best_paths.append(str(best_path))

    df_sum = pd.DataFrame(fold_logs)
    print("\n===== K-Fold Summary =====")
    print(df_sum)
    print(f"{SELECT_METRIC} mean ± std: {df_sum[SELECT_METRIC].mean():.4f} ± {df_sum[SELECT_METRIC].std():.4f}")
    print("AUC mean ± std: {:.4f} ± {:.4f}".format(df_sum["AUC"].mean(), df_sum["AUC"].std()))
    df_sum.to_csv(SAVE_DIR / "kfold_summary.csv", index=False, encoding="utf-8")
    print("Saved k-fold summary to:", SAVE_DIR / "kfold_summary.csv")

    df_all = pd.DataFrame(history_all)
    df_all.to_csv(SAVE_DIR / "kfold_history.csv", index=False, encoding="utf-8")
    print("Saved all-fold history to:", SAVE_DIR / "kfold_history.csv")

    print("Best checkpoints:", best_paths)

if __name__ == "__main__":
    main()
