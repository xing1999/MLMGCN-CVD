import os, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

SELF_LOOP_EPS = 0.10

# ==============================
# I/O & utils
# ==============================
def _load_np_any(path: Path):
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if "emb" in z: return z["emb"]
        if "tok" in z: return z["tok"]
        if "shape" in z: return z["shape"]
        if "indices" in z and "weights" in z: return (z["indices"], z["weights"])
        keys = list(z.keys())
        return z[keys[0]]
    return np.load(path)

def _fix_knn_pair(indices, weights, L_ref):
    idx = np.asarray(indices)
    wts = np.asarray(weights)

    while idx.ndim > 2 and 1 in idx.shape:
        idx = np.squeeze(idx)
    while wts.ndim > 2 and 1 in wts.shape:
        wts = np.squeeze(wts)

    def _reshape_to_2d(arr):
        if arr.ndim == 1:
            if arr.size % L_ref != 0:
                return arr.reshape(1, -1)
            K = arr.size // L_ref
            return arr.reshape(L_ref, K)
        if arr.ndim == 2:
            r, c = arr.shape
            # [L,K] / [K,L] / [1,L*K] / [L*K,1]
            if r == L_ref:      return arr
            if c == L_ref:      return arr.T
            if r == 1:          return arr
            if c == 1 and (r % L_ref == 0):
                return arr.reshape(r // L_ref, L_ref).T
            return arr
        if arr.ndim >= 3:
            flat = arr.reshape(arr.shape[0], -1)
            r, c = flat.shape
            if r == L_ref: return flat
            if c == L_ref: return flat.T
            if r == 1:     return flat
            return flat

    idx = _reshape_to_2d(idx)
    wts = _reshape_to_2d(wts)

    if idx.shape[0] == 1 and L_ref > 1:
        idx = np.repeat(idx, L_ref, axis=0)
    if wts.shape[0] == 1 and L_ref > 1:
        wts = np.repeat(wts, L_ref, axis=0)

    if idx.shape[0] != L_ref and idx.shape[1] == L_ref:
        idx = idx.T
    if wts.shape[0] != L_ref and wts.shape[1] == L_ref:
        wts = wts.T

    if idx.shape[0] < L_ref:
        need = L_ref - idx.shape[0]
        idx = np.concatenate([idx, np.repeat(idx[-1:, :], need, axis=0)], axis=0)
    elif idx.shape[0] > L_ref:
        idx = idx[:L_ref]

    if wts.shape[0] < L_ref:
        need = L_ref - wts.shape[0]
        wts = np.concatenate([wts, np.repeat(wts[-1:, :], need, axis=0)], axis=0)
    elif wts.shape[0] > L_ref:
        wts = wts[:L_ref]

    K = min(idx.shape[1], wts.shape[1])
    idx = idx[:, :K]
    wts = wts[:, :K]

    idx = idx.astype(np.int64, copy=False)
    wts = wts.astype(np.float32, copy=False)

    idx = np.where((idx >= 0) & (idx < L_ref), idx, np.arange(L_ref)[:, None])
    wts = np.where(np.isfinite(wts) & (wts > 0), wts, 0.0)

    row_sum = wts.sum(axis=1, keepdims=True)
    zero_row = (row_sum <= 1e-12).reshape(-1)
    if np.any(zero_row):
        wts[zero_row, :] *= 0.0
        idx[zero_row, :] = np.tile(np.arange(L_ref).reshape(-1, 1), (1, K))[zero_row]
        wts[zero_row, 0] = 1.0
        row_sum = wts.sum(axis=1, keepdims=True)
    wts = wts / (row_sum + 1e-8)
    return idx, wts

# ==============================
# Dataset (switchable structure branch)
# ==============================
class NPZSeqDataset(Dataset):
    def __init__(self, labels_csv, feat_root, struct_mode="ss", use_nt_seq=True):
        assert struct_mode in ("ss", "shape"), "struct_mode 仅支持 'ss' 或 'shape'"
        self.struct_mode = struct_mode
        self.use_nt_seq = use_nt_seq

        df = pd.read_csv(labels_csv)
        assert "label" in df.columns, "labels.csv 必须包含 'label' 列"
        self.labels = df["label"].astype(int).tolist()
        self.root = Path(feat_root)

        if struct_mode == "ss":
            ss_dir = self.root / "ss_struct"
            if not ss_dir.exists():
                raise FileNotFoundError(f"未找到单链结构目录：{ss_dir}")
        else:
            shp_dir = self.root / "dnashape"
            if not shp_dir.exists():
                raise FileNotFoundError(f"未找到 DNAshape 目录：{shp_dir}")

    def __len__(self):
        return len(self.labels)

    # ---- helpers ----
    def _load_shape_feat(self, sid: str) -> np.ndarray:
        p_shape_npy = (self.root/"dnashape"/sid).with_suffix(".npy")
        p_shape_npz = (self.root/"dnashape"/sid).with_suffix(".npz")
        if p_shape_npy.exists():
            arr = _load_np_any(p_shape_npy).astype(np.float32)   # [L,4]
        elif p_shape_npz.exists():
            arr = _load_np_any(p_shape_npz).astype(np.float32)   # key='shape'
        else:
            raise FileNotFoundError(f"DNAshape not found for {sid}")
        return arr

    def _load_ss_feat(self, sid: str) -> np.ndarray:
        p_ss = (self.root/"ss_struct"/sid).with_suffix(".npz")
        if not p_ss.exists():
            raise FileNotFoundError(f"ss_struct not found for {sid}")
        arr = np.load(p_ss, allow_pickle=False)["feats"].astype(np.float32)  # [L, C_ss]
        return arr

    def __getitem__(self, idx):
        sid = f"{idx:06d}"

        # 1) DNABERT-S
        p_s = (self.root/"dnaberts_emb"/sid).with_suffix(".npz")
        s_emb = _load_np_any(p_s).astype(np.float32)      # [L_s, Ds]

        # 2) NT token
        p_nt = (self.root/"nt_emb_tok"/sid).with_suffix(".npz")
        nt_emb = _load_np_any(p_nt).astype(np.float32)    # [L_nt, Dn]

        # 2') NT sequence-level
        if self.use_nt_seq:
            p_nt_seq = (self.root/"nt_emb_seq"/sid).with_suffix(".npy")
            if p_nt_seq.exists():
                nt_seq = np.load(p_nt_seq).astype(np.float32)   # [Dn]
            else:

                nt_seq = nt_emb.mean(axis=0).astype(np.float32) # [Dn]
        else:

            nt_seq = np.zeros((nt_emb.shape[1],), dtype=np.float32)

        # 3) ss_struct  or  DNAshape（
        if self.struct_mode == "ss":
            shp = self._load_ss_feat(sid)       # [L_ss, C_ss]
        else:
            shp = self._load_shape_feat(sid)    # [L_sh, 4]

        # 4) attention KNN
        p_att = (self.root/"dnaberts_attn"/sid).with_suffix(".npz")
        indices, weights = _load_np_any(p_att)

        L = min(s_emb.shape[0], nt_emb.shape[0], shp.shape[0])
        s_emb   = s_emb[:L]
        nt_emb  = nt_emb[:L]
        shp     = shp[:L]
        indices, weights = _fix_knn_pair(indices, weights, L)

        y = int(self.labels[idx])
        return sid, s_emb, nt_emb, shp, (indices, weights), nt_seq, y

def collate_identity(batch):
    return batch
#
# class NPZSeqDataset(Dataset):
#     def __init__(self, labels_csv, feat_root):
#         df = pd.read_csv(labels_csv)
#         assert "label" in df.columns, "labels.csv 必须包含 'label' 列"
#         self.labels = df["label"].astype(int).tolist()
#         self.root = Path(feat_root)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         sid = f"{idx:06d}"
#         # 1) DNABERT-S
#         p_s = (self.root/"dnaberts_emb"/sid).with_suffix(".npz")
#         s_emb = _load_np_any(p_s).astype(np.float32)      # [L_s, Ds]
#         # 2) NT token
#         p_nt = (self.root/"nt_emb_tok"/sid).with_suffix(".npz")
#         nt_emb = _load_np_any(p_nt).astype(np.float32)    # [L_nt, Dn]
#         # 2') NT sequence-level
#         p_nt_seq = (self.root/"nt_emb_seq"/sid).with_suffix(".npy")
#         if p_nt_seq.exists():
#             nt_seq = np.load(p_nt_seq).astype(np.float32) # [Dn]
#         else:
#             nt_seq = nt_emb.mean(axis=0).astype(np.float32)
#         # 3) DNAshape
#         p_shape_npy = (self.root/"dnashape"/sid).with_suffix(".npy")
#         p_shape_npz = (self.root/"dnashape"/sid).with_suffix(".npz")
#         if p_shape_npy.exists():
#             shp = _load_np_any(p_shape_npy).astype(np.float32)   # [L_sh,4]
#         elif p_shape_npz.exists():
#             shp = _load_np_any(p_shape_npz).astype(np.float32)   # key='shape'
#         else:
#             raise FileNotFoundError(f"DNAshape not found for {sid}")
#         # 4) attention KNN
#         p_att = (self.root/"dnaberts_attn"/sid).with_suffix(".npz")
#         indices, weights = _load_np_any(p_att)
#
#
#         L = min(s_emb.shape[0], nt_emb.shape[0], shp.shape[0])
#         s_emb = s_emb[:L]; nt_emb = nt_emb[:L]; shp = shp[:L]
#         indices, weights = _fix_knn_pair(indices, weights, L)
#
#         y = int(self.labels[idx])
#         return sid, s_emb, nt_emb, shp, (indices, weights), nt_seq, y
#
# def collate_identity(batch):
#     return batch

# ==============================
# Layers
# ==============================
class BranchAlign(nn.Module):
    def __init__(self, d_in, d_out=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Identity() if (d_out is None or d_out == d_in) else nn.Linear(d_in, d_out)
    def forward(self, x):

        return self.proj(self.norm(x))

class ShapeEncoder(nn.Module):
    """DNAshape -- Transformer (batch_first=True)"""
    def __init__(self, dim=4, num_layers=2, nhead=2, ff=64, dropout=0.1):
        super().__init__()
        self.ln_in = nn.LayerNorm(dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fnn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):

        if x.dim() == 2:
            x = x.unsqueeze(0); squeeze_back = True
        elif x.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(f"ShapeEncoder expects [L,C] or [B,L,C], got {tuple(x.shape)}")
        x = self.ln_in(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.fnn(x) + x
        x = self.ln_out(x)
        if squeeze_back:
            x = x.squeeze(0)
        return x

class ContentAdjMasked(nn.Module):
    def __init__(self, dim, tau=0.2, init_beta=0.3):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.tau   = tau
        self.mix   = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

    def forward(self, H, knn_indices, knn_weights, self_loop_eps=0.0):
        if not torch.is_tensor(knn_indices):
            knn_indices = torch.as_tensor(knn_indices, device=H.device)
        else:
            knn_indices = knn_indices.to(H.device)
        if not torch.is_tensor(knn_weights):
            knn_weights = torch.as_tensor(knn_weights, device=H.device, dtype=torch.float32)
        else:
            knn_weights = knn_weights.to(H.device, dtype=torch.float32)

        while knn_indices.dim() > 2 and 1 in knn_indices.shape:
            knn_indices = knn_indices.squeeze()
        while knn_weights.dim() > 2 and 1 in knn_weights.shape:
            knn_weights = knn_weights.squeeze()

        L = H.size(0)
        if knn_indices.dim() == 1:
            if knn_indices.numel() % L != 0:
                knn_indices = knn_indices.view(1, -1)
                knn_weights = knn_weights.view(1, -1)
            else:
                K = knn_indices.numel() // L
                knn_indices = knn_indices.view(L, K)
                knn_weights = knn_weights.view(L, K)
        elif knn_indices.dim() == 2:
            r, c = knn_indices.shape
            if r != L and c == L:
                knn_indices = knn_indices.t()
                knn_weights = knn_weights.t()
            elif r == 1 and (c >= 1):
                knn_indices = knn_indices.repeat(L, 1)
                knn_weights = knn_weights.repeat(L, 1)
            elif c == 1 and (r % L == 0):
                K = r // L
                knn_indices = knn_indices.view(K, L).t()
                knn_weights = knn_weights.view(K, L).t()
        else:
            raise ValueError(f"KNN must be 1D/2D after squeeze, got {tuple(knn_indices.shape)}")

        if knn_indices.size(0) != L:
            if knn_indices.size(0) < L:
                need = L - knn_indices.size(0)
                knn_indices = torch.cat([knn_indices, knn_indices[-1:].repeat(need, 1)], dim=0)
                knn_weights = torch.cat([knn_weights, knn_weights[-1:].repeat(need, 1)], dim=0)
            else:
                knn_indices = knn_indices[:L]
                knn_weights = knn_weights[:L]

        L, K = knn_indices.shape
        Q = self.query(H)   # [L,d]
        Kmat = self.key(H)  # [L,d]
        rows = torch.arange(L, device=H.device).unsqueeze(1).expand(L, K)
        cols = knn_indices
        sim  = (Q[rows] * Kmat[cols]).sum(dim=-1)          # [L,K]
        A_soft = torch.softmax(sim / self.tau, dim=-1)     # [L,K]
        beta = torch.sigmoid(self.mix)
        w_fixed = knn_weights.float()
        w_mix = (1 - beta) * w_fixed + beta * A_soft
        w_mix = w_mix / (w_mix.sum(dim=1, keepdim=True) + 1e-8)
        if self_loop_eps > 0.0:
            rows_flat = torch.cat([rows.reshape(-1), torch.arange(L, device=H.device)])
            cols_flat = torch.cat([cols.reshape(-1), torch.arange(L, device=H.device)])
            vals_flat = torch.cat([w_mix.reshape(-1), torch.full((L,), float(self_loop_eps), device=H.device)])
        else:
            rows_flat = rows.reshape(-1); cols_flat = cols.reshape(-1); vals_flat = w_mix.reshape(-1)
        A = torch.sparse_coo_tensor(torch.vstack([rows_flat, cols_flat]), vals_flat, size=(L, L)).coalesce()
        return A

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super().__init__()
        self.variant = variant
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x, adj, h0, lamda, alpha, l):
        theta = min(1, math.log(lamda / l + 1))
        if adj.is_sparse:
            hi = torch.sparse.mm(adj, x)
        else:
            hi = torch.mm(adj, x)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        out = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            out = out + x
        return out

class DeepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, dropout, lamda, alpha, variant):
        super().__init__()
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.convs = nn.ModuleList([GraphConvolution(nhidden, nhidden, residual=True, variant=variant)
                                    for _ in range(nlayers)])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.post_ln = nn.LayerNorm(nhidden)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training).float()
        h0 = self.act(self.fc_in(x))
        h  = h0
        for i, conv in enumerate(self.convs, start=1):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act(conv(h, adj, h0, self.lamda, self.alpha, i))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.post_ln(h)

class QueryAttnReadout(nn.Module):
    """Q learnable queries -- H"""
    def __init__(self, dim, num_queries=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_queries, dim) * (1.0 / math.sqrt(dim)))
        self.scale = dim ** 0.5
        self.do = nn.Dropout(dropout)
    def forward(self, H):  # H:[L,dim]
        att = torch.matmul(self.q, H.T) / self.scale   # [Q, L]
        alpha = torch.softmax(att, dim=-1)             # [Q,L]
        Z = torch.matmul(alpha, H)                     # [Q, dim]
        Z = self.do(Z)
        return Z.mean(0)                               # [dim]

class ConvRefine1D(nn.Module):

    def __init__(self, dim, ks_list=(3,), dilations=None, dropout=0.1):
        super().__init__()
        if dilations is None:
            dilations = [1] * len(ks_list)
        assert len(dilations) == len(ks_list)
        layers = []
        for k, d in zip(ks_list, dilations):
            pad = (k // 2) * d
            layers += [
                nn.Conv1d(dim, dim, kernel_size=k, padding=pad, dilation=d, groups=dim, bias=False),
                nn.Conv1d(dim, dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.ln  = nn.LayerNorm(dim)

    def forward(self, x):
        squeeze_back = False
        if x.dim() == 2:   # [L,C]
            x = x.unsqueeze(0); squeeze_back = True
        x0 = x
        x  = x.transpose(1, 2)   # [B,L,C] -> [B,C,L]
        y  = self.net(x)
        y  = y.transpose(1, 2)   # [B,C,L] -> [B,L,C]
        y  = self.ln(y)
        y  = y + x0
        if squeeze_back:
            y = y.squeeze(0)
        return y

# ==============================
# Model
# ==============================
class SeqClassifier(nn.Module):
    def __init__(self, nlayers, dim_s, dim_nt, dim_shape, hidden_dim, dropout, lamda, alpha, variant,
                 use_content_masked=True, num_queries=4):
        super().__init__()
        self.shape_encoder = ShapeEncoder(dim=dim_shape, num_layers=2, nhead=2, ff=64, dropout=dropout)
        self.align_s   = BranchAlign(dim_s)
        self.align_nt  = BranchAlign(dim_nt)
        self.align_shp = BranchAlign(dim_shape)

        in_dim = dim_s + dim_nt + dim_shape

        self.gcn = DeepGCN(nlayers, in_dim, hidden_dim, dropout, lamda, alpha, variant)
        self.use_content_masked = use_content_masked
        if use_content_masked:
            self.cadj_masked = ContentAdjMasked(dim=hidden_dim, tau=0.2, init_beta=0.3)

        # >>> CNN
        self.cnn_post = ConvRefine1D(hidden_dim, ks_list=(3, 3), dilations=[1, 2], dropout=dropout)

        self.readout = QueryAttnReadout(hidden_dim, num_queries=num_queries, dropout=dropout)

        # ===== nt_seq =====
        self.nt_seq_proj = nn.Sequential(
            nn.LayerNorm(dim_nt),
            nn.Linear(dim_nt, hidden_dim),
            nn.Tanh()
        )
        self.nt_gate = nn.Sequential(
            nn.LayerNorm(dim_nt),
            nn.Linear(dim_nt, hidden_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(256, 64),             nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    # -------- 单样本前向 --------
    def _forward_one(self, s_emb, nt_emb, shp, knn_indices, knn_weights, nt_seq, shp_kpm=None):
        #  shape
        if shp_kpm is not None:
            shp_enc = self.shape_encoder(shp, key_padding_mask=shp_kpm.unsqueeze(0)).squeeze(0)
        else:
            shp_enc = self.shape_encoder(shp)

        s_emb   = self.align_s(s_emb)     # [L, Ds]
        nt_emb  = self.align_nt(nt_emb)   # [L, Dn]
        shp_enc = self.align_shp(shp_enc) # [L, dim_shape]
        node = torch.cat([s_emb, nt_emb, shp_enc], dim=1)  # [L, in_dim]

        h0 = self.gcn.fc_in(F.dropout(node, p=self.gcn.dropout, training=self.training)).relu()
        if self.use_content_masked:
            adj = self.cadj_masked(h0, knn_indices, knn_weights, self_loop_eps=SELF_LOOP_EPS)
        else:
            idx_np = knn_indices if isinstance(knn_indices, np.ndarray) else knn_indices.detach().cpu().numpy()
            wts_np = knn_weights if isinstance(knn_weights, np.ndarray) else knn_weights.detach().cpu().numpy()
            L, K = idx_np.shape
            row = np.repeat(np.arange(L, dtype=np.int64), K)
            col = idx_np.astype(np.int64).reshape(-1)
            val = wts_np.astype(np.float32).reshape(-1)
            if SELF_LOOP_EPS > 0.0:
                row = np.concatenate([row, np.arange(L, dtype=np.int64)])
                col = np.concatenate([col, np.arange(L, dtype=np.int64)])
                val = np.concatenate([val, np.full((L,), float(SELF_LOOP_EPS), dtype=np.float32)])
            ij = torch.tensor([row, col], device=node.device)
            vv = torch.tensor(val, device=node.device)
            adj = torch.sparse_coo_tensor(ij, vv, size=(L, L)).coalesce()

        # 4) GCN → CNN
        h = self.gcn(node, adj)        # [L, hidden]
        h = self.cnn_post(h)           # [L, hidden]


        g = self.readout(h)            # [hidden]
        z_seq = self.nt_seq_proj(nt_seq)   # [hidden]
        gate  = self.nt_gate(nt_seq)       # [hidden]
        g_fused = torch.cat([g * (1 + gate), z_seq], dim=-1)  # [2*hidden]

        return self.head(g_fused).squeeze(-1)


    def forward(self, s_emb, nt_emb, shp, knn_indices, knn_weights, nt_seq, mask=None):

        if s_emb.dim() == 2:
            shp_kpm = None
            if mask is not None and 'shp' in mask:
                m = mask['shp']
                if m.dim() == 1:
                    shp_kpm = (m == 0)
                elif m.dim() == 2 and m.size(0) == 1:
                    shp_kpm = (m[0] == 0)
            return self._forward_one(s_emb, nt_emb, shp, knn_indices, knn_weights, nt_seq, shp_kpm)


        assert s_emb.dim() == 3 and nt_emb.dim() == 3 and shp.dim() == 3, "Batch mode expects [B,L,C]"
        B = s_emb.size(0)
        assert isinstance(knn_indices, (list, tuple)) and isinstance(knn_weights, (list, tuple)) \
               and len(knn_indices) == B and len(knn_weights) == B, \
               "Batch mode requires per-sample KNN lists of length B"
        assert nt_seq.dim() == 2 and nt_seq.size(0) == B, "Batch nt_seq must be [B,D]"

        logits = []
        for b in range(B):
            s_b   = s_emb[b]
            nt_b  = nt_emb[b]
            shp_b = shp[b]
            ntq_b = nt_seq[b]
            idx_b = knn_indices[b]
            wts_b = knn_weights[b]

            shp_kpm = None
            if mask is not None and 'shp' in mask:
                m = mask['shp']
                if m.dim() == 2:      # [B,L]
                    shp_kpm = (m[b] == 0)
                elif m.dim() == 1:    # [L]
                    shp_kpm = (m == 0)

            logit_b = self._forward_one(s_b, nt_b, shp_b, idx_b, wts_b, ntq_b, shp_kpm)
            logits.append(logit_b)

        return torch.stack(logits, dim=0)  # [B]



