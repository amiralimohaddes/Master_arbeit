#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 fileformat=unix :
"""
EAT + KMeansNN anomaly pipeline with optional LoRA fine-tuning (PEFT)
(v3-report): masked pooling, last-N LoRA targeting, standardized-space loss, optional CosFace,
and a comprehensive diagnostics/report generator (plots + Excel).

Run examples
------------
# Baseline (no LoRA) + full report
python eat_kmeans_lora_pipeline_v3_report.py --dataset_root ./dataset --machine bearing --make_report

# With LoRA (auto targets, last 6 blocks) + full report
python eat_kmeans_lora_pipeline_v3_report.py --dataset_root ./dataset --machine bearing \
  --do_lora_ft --auto_hparam --lora_last_n 6 --make_report --use_cosface
"""
from __future__ import annotations

import os
from pathlib import Path
import sys, math, json, argparse, time, re, random
from typing import List, Optional, Tuple, Dict

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import pandas as pd

from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, roc_curve, auc
)

# Optional libs
HAVE_UMAP = False
try:
    import umap  # pip install umap-learn
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

HAVE_SCIPY = False
try:
    from scipy.linalg import sqrtm
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----- project utilities (expected available) -----
from load_dataset import load_data
from metrics import compute_and_save_metrics

# ---------------- Defaults ----------------
MODEL_ID        = "worstchan/EAT-base_epoch30_pretrain"
SR              = 16000
TARGET_LENGTH   = 1024
N_CLUSTERS      = 18
BATCH_SIZE      = 32
MAX_TRAIN       = 10000
SEED            = 42

# LoRA defaults
DEFAULT_LORA_R         = 32
DEFAULT_LORA_ALPHA     = 64
DEFAULT_LORA_DROPOUT   = 0.05
DEFAULT_FT_LR          = 1e-4
DEFAULT_FT_EPOCHS      = 50

# EAT fbank normalization
NORM_MEAN = -4.268
NORM_STD  = 4.569

# -------------- Helpers ---------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def infer_domain_from_path(p) -> str:
    # Heuristic first (fast path)
    s = str(p); s_lower = s.lower(); name = Path(s).stem.lower()
    if ("source" in name) or ("src" in name) or "/source/" in s_lower or "\\source\\" in s_lower:
        return "source"
    if ("target" in name) or ("tgt" in name) or "/target/" in s_lower or "\\target\\" in s_lower:
        return "target"

    global _META_DOMAIN_CACHE, _META_LABEL_CACHE, _META_LOADED_DIRS
    try:
        _META_DOMAIN_CACHE
    except NameError:
        _META_DOMAIN_CACHE = {}
        _META_LABEL_CACHE = {}
        _META_LOADED_DIRS = set()

    path_obj = Path(p)
    fname = path_obj.name

    def try_load_metadata(root_dir: Path):
        import csv
        if root_dir in _META_LOADED_DIRS:
            return
        attr_csv = root_dir / "attributes_00.csv"
        if attr_csv.exists():
            try:
                with attr_csv.open("r", encoding="utf-8") as f:
                    first = f.readline()
                    header = [h.strip().lower() for h in first.strip().split(',')]
                    # If header only contains file_name (common), infer domain from each subsequent line
                    if header == ["file_name"]:
                        for line in f:
                            fn = line.strip().split(',')[0].strip()
                            if not fn:
                                continue
                            low = fn.lower()
                            if "source" in low:
                                _META_DOMAIN_CACHE[fn] = "source"
                            elif "target" in low:
                                _META_DOMAIN_CACHE[fn] = "target"
                    else:
                        # General case: use DictReader for possible explicit domain column
                        f.seek(0)
                        reader = csv.DictReader(f)
                        for row in reader:
                            if not row:
                                continue
                            keys_lower = {k.lower(): v for k, v in row.items() if k}
                            fkey = None
                            for cand in ("filename", "file_name", "name", "file"):
                                if cand in keys_lower:
                                    fkey = cand; break
                            if not fkey:
                                continue
                            dom_val = None
                            # explicit domain columns
                            for cand in ("domain", "dom", "src_tgt", "source_target"):
                                if cand in keys_lower and keys_lower[cand]:
                                    dom_val = keys_lower[cand].strip().lower(); break
                            if dom_val in ("source", "target"):
                                _META_DOMAIN_CACHE[keys_lower[fkey]] = dom_val
                            else:
                                low = keys_lower[fkey].lower()
                                if "source" in low:
                                    _META_DOMAIN_CACHE[keys_lower[fkey]] = "source"
                                elif "target" in low:
                                    _META_DOMAIN_CACHE[keys_lower[fkey]] = "target"
            except Exception:
                pass
        _META_LOADED_DIRS.add(root_dir)

    for parent in [path_obj.parent] + list(path_obj.parents):
        if parent in _META_LOADED_DIRS:
            break
        if (parent / "attributes_00.csv").exists():
            try_load_metadata(parent)
            break

    dom = _META_DOMAIN_CACHE.get(fname) or _META_DOMAIN_CACHE.get(fname.lower())
    if dom in ("source", "target"):
        return dom
    return "unknown"


def infer_label_from_path(p) -> int:
    """Heuristic label inference with CSV fallback (ground_truth & attributes)."""
    s = str(p).lower()
    if any(k in s for k in ["anomaly", "abnormal", "anomal"]):
        return 1
    if "normal" in s:
        return 0

    global _META_DOMAIN_CACHE, _META_LABEL_CACHE, _META_LOADED_DIRS
    try:
        _META_LABEL_CACHE
    except NameError:
        _META_DOMAIN_CACHE = {}
        _META_LABEL_CACHE = {}
        _META_LOADED_DIRS = set()

    path_obj = Path(p)
    fname = path_obj.name

    def try_load_metadata(root_dir: Path):
        import csv
        if root_dir in _META_LOADED_DIRS:
            return
        # Reuse domain inference for attributes (adds domain cache + maybe label tokens)
        attr_csv = root_dir / "attributes_00.csv"
        if attr_csv.exists():
            try:
                with attr_csv.open("r", encoding="utf-8") as f:
                    first = f.readline()
                    header = [h.strip().lower() for h in first.strip().split(',')]
                    if header == ["file_name"]:
                        for line in f:
                            fn = line.strip().split(',')[0].strip()
                            if not fn:
                                continue
                            low = fn.lower()
                            if "source" in low:
                                _META_DOMAIN_CACHE[fn] = "source"
                            elif "target" in low:
                                _META_DOMAIN_CACHE[fn] = "target"
                    else:
                        f.seek(0)
                        reader = csv.DictReader(f)
                        for row in reader:
                            if not row:
                                continue
                            keys_lower = {k.lower(): v for k, v in row.items() if k}
                            fkey = None
                            for cand in ("filename", "file_name", "name", "file"):
                                if cand in keys_lower:
                                    fkey = cand; break
                            if not fkey:
                                continue
                            low = keys_lower[fkey].lower()
                            if "source" in low:
                                _META_DOMAIN_CACHE[keys_lower[fkey]] = "source"
                            elif "target" in low:
                                _META_DOMAIN_CACHE[keys_lower[fkey]] = "target"
            except Exception:
                pass
        # ground truth CSV(s) – handle headerless two-column format filename,label
        for cand in list(root_dir.glob("ground_truth_*_test.csv")) + list(root_dir.glob("ground_truth*.csv")):
            if not cand.exists():
                continue
            try:
                with cand.open("r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if not lines:
                    continue
                # Detect header vs data: if first line contains .wav and a comma -> treat all lines as data
                first = lines[0]
                header_like = ('.wav' not in first.lower())
                data_start = 1 if header_like else 0
                # If header-like but only two columns and first col ends with .wav, revert
                if header_like:
                    parts = [p.strip() for p in first.split(',')]
                    if len(parts) == 2 and parts[0].lower().endswith('.wav'):
                        data_start = 0
                for ln in lines[data_start:]:
                    parts = [p.strip() for p in ln.split(',')]
                    if len(parts) < 2:
                        continue
                    fn, lab = parts[0], parts[1].lower()
                    if not fn:
                        continue
                    if lab in ("1", "anomaly", "anomalous", "abnormal", "true", "yes"):
                        _META_LABEL_CACHE[fn] = 1
                    elif lab in ("0", "normal", "false", "no"):
                        _META_LABEL_CACHE[fn] = 0
            except Exception:
                continue
        _META_LOADED_DIRS.add(root_dir)

    for parent in [path_obj.parent] + list(path_obj.parents):
        if parent in _META_LOADED_DIRS:
            break
        if (parent / "attributes_00.csv").exists() or list(parent.glob("ground_truth_*_test.csv")) or list(parent.glob("ground_truth*.csv")):
            try_load_metadata(parent)
            break

    if fname in _META_LABEL_CACHE:
        return _META_LABEL_CACHE[fname]
    if fname.lower() in _META_LABEL_CACHE:
        return _META_LABEL_CACHE[fname.lower()]

    return 0

def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ---------- EAT feature extractor ----------
class EATFeatureExtractor:
    def __init__(self, model_id: str = MODEL_ID, sr: int = SR, target_length: Optional[int] = TARGET_LENGTH,
                 train_mode: bool = False, device: Optional[torch.device] = None, pool_mode: str="cls_only"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.sr     = sr
        self.tlen   = target_length
        self.train_mode = train_mode
        self.pool_mode = pool_mode
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

    def _tokens_forward(self, fbank_batch: torch.Tensor) -> torch.Tensor:
        """
        Always call the model's feature extractor and ensure inputs
        are on the SAME device as the model's weights.
        Returns tokens as [B, L, D].
        """
        # 1) Align device with the model
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            # no parameters? fall back to self.device
            model_device = getattr(self, "device", fbank_batch.device)
        if fbank_batch.device != model_device:
            fbank_batch = fbank_batch.to(model_device)

        # 2) Call the proper extractor (do NOT fall back to __call__/forward)
        if hasattr(self.model, "extract_features"):
            seq = self.model.extract_features(fbank_batch)
        else:
            # If your remote code uses another name (e.g., forward_features), call it here:
            raise RuntimeError(
                "Backbone does not expose `extract_features`. "
                "Please update the model wrapper to call its feature extractor method."
            )

        # 3) Canonicalize to [B, L, D]
        if isinstance(seq, (list, tuple)):
            seq = seq[0]
        if seq.dim() == 4:  # [B,1,L,D]
            seq = seq.squeeze(1)
        return seq
    
    def _wav_to_fbank(self, wav: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        Returns normalized fbank [T,128] (padded/clipped to target_length) AND the valid length L<=T.
        """
        w = torch.tensor(wav, dtype=torch.float32)
        if w.ndim != 1:
            w = w.view(-1)
        w = w - w.mean()
        fb = torchaudio.compliance.kaldi.fbank(
            w.unsqueeze(0), htk_compat=True, sample_frequency=self.sr,
            use_energy=False, window_type="hanning", num_mel_bins=128,
            dither=0.0, frame_shift=10,
        )  # [T,128]
        T_true = fb.size(0)
        if self.tlen is not None:
            if T_true < self.tlen:
                fb = F.pad(fb, (0, 0, 0, self.tlen - T_true))
            elif T_true > self.tlen:
                fb = fb[: self.tlen, :]
        fb = (fb - NORM_MEAN) / (NORM_STD * 2)  # FIX: reintroduce *2 scaling
        fb = fb.transpose(-1, -2) # [128,T]
        return fb, min(T_true, self.tlen or T_true)

    def _masked_mean(self, seq: torch.Tensor, lengths: torch.Tensor, drop_cls: bool=True) -> torch.Tensor:
        # unchanged helper, used only for non-CLS modes
        if drop_cls and seq.size(1) >= 2:
            seq = seq[:, 1:, :]
        B, T, D = seq.shape
        mask = (torch.arange(T, device=seq.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
        return (seq * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    @torch.no_grad()
    def encode_batch(self, wavs: List[np.ndarray]) -> np.ndarray:
        # build fbank batch as before
        fb_list, lens = [], []
        for w in wavs:
            fb, L = self._wav_to_fbank(w)
            fb_list.append(fb)
            lens.append(L)
        x = torch.stack(fb_list, dim=0).unsqueeze(1).to(self.device)  # [B,1,128,T]
        lens = torch.tensor(lens, dtype=torch.long, device=self.device)

        seq = self._tokens_forward(x)  # [B,L,D]

        if self.pool_mode == "cls_only":
            cls = seq[:, 0, :]  # <-- CLS
            return cls.detach().cpu().numpy()
        elif self.pool_mode == "mean_no_cls":
            pooled = self._masked_mean(seq, lengths=lens, drop_cls=True)
            return pooled.detach().cpu().numpy()
        else:  # "mean_with_cls"
            pooled = self._masked_mean(seq, lengths=lens, drop_cls=False)
            return pooled.detach().cpu().numpy()

    def forward_cls(self, fbank_batch: torch.Tensor, lengths: Optional[torch.Tensor]=None) -> torch.Tensor:
        seq = self._tokens_forward(fbank_batch)  # [B,L,D]
        if self.pool_mode == "cls_only":
            return seq[:, 0, :]              # <-- CLS during FT
        elif self.pool_mode == "mean_no_cls":
            return self._masked_mean(seq, lengths=lengths.to(seq.device), drop_cls=True)
        else:
            return self._masked_mean(seq, lengths=lengths.to(seq.device), drop_cls=False)

# ----------------- LoRA utilities -----------------
ATTN_TOKENS = [
    "qkv", "proj",
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "query", "key", "value", "in_proj", "in_proj_weight", "in_proj_bias",
]

def present_attention_tokens(model: nn.Module) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    present = [tok for tok in ATTN_TOKENS if any(tok in n for n in names)]
    return sorted(set(present), key=present.index)

def suggest_attention_like_names(model: nn.Module, limit: int = 40) -> List[str]:
    names = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))]
    attnish = [n for n in names if any(k in n.lower() for k in ["attn", "attention", "q", "k", "v", "proj", "qkv"]) ]
    attnish_sorted = sorted(attnish, key=lambda s: (len(s), s))
    return attnish_sorted[:limit]

def resolve_lora_targets(model: nn.Module, override: Optional[str]) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    if override:
        raw = [t.strip() for t in override.split(',') if t.strip()]
        filtered = [t for t in raw if any(t in n for n in names)]
        if filtered:
            return sorted(set(filtered), key=filtered.index)
        print(f"[LoRA-FT][WARN] None of --lora_targets matched actual module names: {raw}")
    present = present_attention_tokens(model)
    if present:
        return present
    fallback = ["qkv", "q_proj", "k_proj", "v_proj"]
    fb_filtered = [t for t in fallback if any(t in n for n in names)]
    if fb_filtered:
        return fb_filtered
    suggestions = suggest_attention_like_names(model)
    raise ValueError(
        "Could not discover LoRA targets in this EAT model. Try passing --lora_targets explicitly.\n"
        f"Examples to try: --lora_targets qkv   or   --lora_targets q_proj,k_proj,v_proj\n"
        + "Here are some attention-like module names I found:\n- " + "\n- ".join(suggestions)
    )

def find_lora_target_names(model: nn.Module, tokens: List[str], last_n: int,
                           include_only_attn: bool = True, include_mlp_fc1: bool = False) -> List[str]:
    # discover block index range
    block_ids = []
    for n, _ in model.named_modules():
        m = re.match(r"^model\.blocks\.(\d+)\.", n)
        if m:
            block_ids.append(int(m.group(1)))
    if not block_ids:
        return []
    max_blk = max(block_ids)
    cutoff = max(0, max_blk - last_n + 1)

    selected: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        m = re.match(r"^model\.blocks\.(\d+)\.", name)
        if not m:
            continue
        blk = int(m.group(1))
        if blk < cutoff:
            continue
        if include_only_attn and ".attn." not in name:
            if include_mlp_fc1 and name.endswith(".mlp.fc1"):
                pass
            else:
                continue
        if not any(tok in name for tok in tokens):
            continue
        selected.append(name)

    seen = set(); out = []
    for n in selected:
        if n not in seen:
            out.append(n); seen.add(n)
    return out

# ---------------- Data for fine-tune --------------
class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, wavs: List[np.ndarray], labels: np.ndarray, sr: int, target_len: Optional[int]):
        assert len(wavs) == len(labels)
        self.wavs = wavs
        self.labels = labels.astype(np.int64)
        self.sr = sr
        self.target_len = target_len

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx: int):
        w = self.wavs[idx]
        y = int(self.labels[idx])
        w_t = torch.tensor(w, dtype=torch.float32)
        if w_t.ndim != 1:
            w_t = w_t.view(-1)
        w_t = w_t - w_t.mean()
        fb = torchaudio.compliance.kaldi.fbank(
            w_t.unsqueeze(0), htk_compat=True, sample_frequency=self.sr,
            use_energy=False, window_type="hanning", num_mel_bins=128,
            dither=0.0, frame_shift=10,
        )  # [T,128]
        T_true = fb.size(0)
        if self.target_len is not None:
            if T_true < self.target_len:
                fb = F.pad(fb, (0, 0, 0, self.target_len - T_true))
            elif T_true > self.target_len:
                fb = fb[: self.target_len, :]
        fb = (fb - NORM_MEAN) / (NORM_STD * 2)  # FIX: reintroduce *2 scaling
        fb = fb.transpose(-1, -2).unsqueeze(0)  # [1,128,T]
        L = min(T_true, self.target_len or T_true)
        return fb.unsqueeze(0), y, L  # [1,T,128], label, valid length

def collate_fb(items):
    fbs, ys, lens = zip(*items)
    x = torch.stack(fbs, dim=0)  # [B,1,128,T]
    y = torch.tensor(ys, dtype=torch.long)
    lengths = torch.tensor(lens, dtype=torch.long)
    return x, y, lengths

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class CosFaceHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, s: float = 16.0, m: float = 0.20):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, n_classes))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        W_n = F.normalize(self.W, dim=0)
        logits = x_n @ W_n  # cosine
        onehot = F.one_hot(y, num_classes=logits.size(1)).float()
        logits_m = logits - self.m * onehot
        return self.s * logits_m

def auto_choose_k(feats: np.ndarray, k_grid: List[int] = None, seed: int = SEED) -> int:
    if k_grid is None:
        k_grid = [12, 18, 24, 32]
    best_k = k_grid[0]
    best_score = -1.0
    X = StandardScaler().fit_transform(feats)
    for k in k_grid:
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
            labels = km.fit_predict(X)
            if len(set(labels)) == 1:
                continue
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1.0
        if sil > best_score:
            best_score = sil
            best_k = k
    return best_k

def count_trainable_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return trainable, total

def compute_cluster_radii(X_std: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    K = centers.shape[0]
    radii = np.zeros(K, dtype=np.float32)
    for k in range(K):
        pts = X_std[labels==k]
        if len(pts)==0:
            radii[k] = 0.0
        else:
            d = np.linalg.norm(pts - centers[k], axis=1)
            radii[k] = np.median(d)  # robust radius
    return radii

# ---------------- Diagnostics utils ----------------
def make_2d(X: np.ndarray, method="umap", seed=42):
    if method == "umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        Z = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    return Z

def plot_2d(Z, labels=None, title="", path=None, centroids=None, point_kwargs=None, legend=True):
    plt.figure(figsize=(6,5))
    labs = np.zeros(len(Z), dtype=int) if labels is None else np.asarray(labels)
    uniq = np.unique(labs)
    for u in uniq:
        m = labs == u
        plt.scatter(Z[m,0], Z[m,1], s=12, alpha=0.8, label=str(u), **(point_kwargs or {}))
    if centroids is not None:
        plt.scatter(centroids[:,0], centroids[:,1], s=140, marker="X", edgecolor="k", linewidths=1.0, label="centroids")
    if legend: plt.legend(markerscale=2, fontsize=8)
    plt.title(title); plt.tight_layout()
    if path: plt.savefig(path, dpi=170); plt.close()

def centroid_distance_heatmap(C: np.ndarray, title: str, path: str):
    import seaborn as sns
    plt.figure(figsize=(5.5,4.8))
    sns.heatmap(np.linalg.norm(C[:,None,:]-C[None,:,:], axis=2), cmap="viridis", square=True)
    plt.title(title); plt.xlabel("centroid"); plt.ylabel("centroid"); plt.tight_layout()
    plt.savefig(path, dpi=170); plt.close()

def silhouette_plot(X_std: np.ndarray, labels: np.ndarray, title: str, path: str):
    svals = silhouette_samples(X_std, labels)
    K = labels.max()+1
    fig, ax = plt.subplots(figsize=(6,5))
    y_lower = 10
    for i in range(K):
        vals = np.sort(svals[labels==i])
        size_i = vals.shape[0]
        y_upper = y_lower + size_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5*size_i, str(i), fontsize=8)
        y_lower = y_upper + 10
    ax.set_title(title); ax.set_xlabel("silhouette value"); ax.set_ylabel("cluster")
    ax.axvline(np.mean(svals), color="red", linestyle="--", linewidth=1)
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()
    return float(np.mean(svals))

from sklearn.metrics import roc_curve, auc
import numpy as np

def pauc_at_fpr(y_true, scores, fpr_max=0.1):
    fpr, tpr, _ = roc_curve(y_true, scores)
    
    # Interpolate to staircase: Repeat each (fpr[i], tpr[i]) and add (fpr[i+1], tpr[i])
    # This handles duplicates by creating horizontal segments
    fpr_stair = [fpr[0]]
    tpr_stair = [tpr[0]]
    for i in range(len(fpr) - 1):
        fpr_stair.extend([fpr[i], fpr[i+1]])
        tpr_stair.extend([tpr[i], tpr[i]])  # Horizontal at tpr[i] until next fpr
    
    fpr_stair = np.array(fpr_stair)
    tpr_stair = np.array(tpr_stair)
    
    mask = fpr_stair <= fpr_max
    if mask.sum() < 2:
        return 0.0, (fpr, tpr)
    
    # Now auc works correctly on the staircased masked points
    partial_auc = auc(fpr_stair[mask], tpr_stair[mask])
    return partial_auc / max(fpr_max, 1e-12), (fpr, tpr)

def plot_roc(y_true, scores, title, path, fpr_max=0.1):
    pA, (fpr, tpr) = pauc_at_fpr(y_true, scores, fpr_max)
    A = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={A:.3f}, pAUC@{fpr_max}={pA:.3f}")
    plt.fill_between(fpr, tpr, where=(fpr<=fpr_max), alpha=0.15)
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()
    return A, pA

def threshold_table(y_true, scores, fprs=(0.005,0.01,0.02,0.05,0.1)):
    fpr, tpr, thr = roc_curve(y_true, scores)
    out = []
    for target in fprs:
        i = np.searchsorted(fpr, target, side="left")
        i = min(i, len(fpr)-1)
        out.append({"target_fpr": target, "tpr": float(tpr[i]), "actual_fpr": float(fpr[i]), "threshold": float(thr[i])})
    return out

def gaussian_stats(X):
    mu = X.mean(0); C = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    return mu, C

def frechet_distance(mu1, C1, mu2, C2):
    if HAVE_SCIPY:
        covmean = sqrtm(C1.dot(C2))
        if np.iscomplexobj(covmean): covmean = covmean.real
        diff = mu1 - mu2
        return float(diff.dot(diff) + np.trace(C1 + C2 - 2*covmean))
    # fallback approximation (no sqrtm): trace(C1+C2) + ||mu1-mu2||^2
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(C1 + C2))

def cdist2(A, B):
    AA = (A*A).sum(1, keepdims=True); BB = (B*B).sum(1, keepdims=True)
    return AA + BB.T - 2*A.dot(B.T)

def mmd_rbf(X, Y, gamma=None):
    if gamma is None: gamma = 1.0 / max(1, X.shape[1])
    XX = np.exp(-gamma * cdist2(X, X))
    YY = np.exp(-gamma * cdist2(Y, Y))
    XY = np.exp(-gamma * cdist2(X, Y))
    return float(XX.mean() + YY.mean() - 2*XY.mean())

def bootstrap_ci(y_true, scores, stat_fn, n=200, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yb = y_true[sample]; sb = scores[sample]
        # Need both classes present
        if len(np.unique(yb)) < 2:
            continue
        vals.append(stat_fn(yb, sb))
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)



# ---------------- ONNX export utility ----------------
import torch
import torch.nn as nn
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def export_seq_bt128_onnx(
    backbone: nn.Module,
    onnx_path: str | Path,
    example_T: int = 1024,
    opset: int = 17,
    drop_cls: bool = False,           # drop leading CLS if present
    match_input_T: bool = False      # if True, interpolate time to match input T
):
    """
    Export with input [B,T,128].
    - If match_input_T=False: output [B,L,D] (tokens), dynamic axes 'tokens'.
    - If match_input_T=True : output [B,T,D] (time interpolated), dynamic axes 'target_length'.

    LoRA is merged if the backbone is a PEFT model.
    """

    # 1) Merge LoRA adapters into base weights (so ONNX has plain weights)
    model = backbone
    if hasattr(model, "merge_and_unload"):
        print("[ONNX] Merging LoRA adapters into base weights…")
        model = model.merge_and_unload()
    else:
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                print("[ONNX] Merging (PeftModel)…")
                model = model.merge_and_unload()
        except Exception:
            pass

    model.eval().to("cpu")

    # 2) Wrapper to adapt [B,T,128] -> [B,1,T,128], handle CLS, and optional resample
    class SeqWrapper(nn.Module):
        def __init__(self, backbone: nn.Module, drop_cls: bool, match_input_T: bool):
            super().__init__()
            self.backbone = backbone
            self.drop_cls = drop_cls
            self.match_input_T = match_input_T

        def forward(self, x_bt128: torch.Tensor) -> torch.Tensor:
            # x_bt128: [B, T, 128]
            B, T, F = x_bt128.shape
            x = x_bt128.unsqueeze(1)  # [B,1,T,128]
            # call backbone
            try:
                seq = self.backbone.extract_features(x)
            except Exception:
                out = self.backbone(x)
                seq = getattr(out, "last_hidden_state", out)
            # seq: usually [B,1,L,D] or [B,L,D]
            if seq.dim() == 4:
                seq = seq.squeeze(1)         # -> [B,L,D]
            # optionally drop CLS token (assume at index 0)
            if self.drop_cls and seq.size(1) >= 2:
                seq = seq[:, 1:, :]          # [B,L-1,D]
            # optionally resample along time to match input T
            if self.match_input_T:
                # interpolate over time: [B,L',D] -> [B,D,L'] -> resize -> [B,T,D]
                y = seq.permute(0, 2, 1)     # [B,D,L']
                y = F.interpolate(y, size=T, mode="linear", align_corners=False)  # [B,D,T]
                y = y.permute(0, 2, 1)       # [B,T,D]
                return y
            else:
                return seq                   # [B,L',D] (tokens)

    wrapper = SeqWrapper(model, drop_cls=drop_cls, match_input_T=match_input_T)

    # 3) Quick sanity forward
    dummy = torch.randn(1, example_T, 128, dtype=torch.float32)
    with torch.no_grad():
        y = wrapper(dummy)
        assert y.dim() == 3, f"Unexpected output rank: {y.shape}"
        # If match_input_T=True, time should equal example_T
        if match_input_T:
            assert y.shape[1] == example_T, f"Output time {y.shape[1]} != input time {example_T}"

    # 4) Export
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if match_input_T:
        dyn_axes = {
            "input":  {0: "batch_size", 1: "target_length"},
            "output": {0: "batch_size", 1: "target_length"},
        }
    else:
        dyn_axes = {
            "input":  {0: "batch_size", 1: "target_length"},
            "output": {0: "batch_size", 1: "tokens"},
        }

    torch.onnx.export(
        wrapper,
        (dummy,),
        f=str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dyn_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[ONNX] Saved: {onnx_path.resolve()}")



# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="EAT + KMeans anomaly with optional LoRA fine-tuning + Report (v3)")
    parser.add_argument("--dataset_root", type=str, default="./dataset")
    parser.add_argument("--machine", type=str, default="bearing")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--target_length", type=int, default=TARGET_LENGTH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_train", type=int, default=MAX_TRAIN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    # LoRA & FT flags
    parser.add_argument("--do_lora_ft", dest="do_lora_ft", action="store_true")
    parser.add_argument("--no_do_lora_ft", dest="do_lora_ft", action="store_false")
    parser.set_defaults(do_lora_ft=False)
    parser.add_argument("--auto_hparam", action="store_true")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora_targets", type=str, default=None, help="Comma-separated substrings, e.g. 'qkv' or 'q_proj,k_proj,v_proj'")
    parser.add_argument("--lora_last_n", type=int, default=4, help="Apply LoRA only to the last N Transformer blocks")
    parser.add_argument("--lora_include_mlp_fc1", action="store_true", help="Also apply LoRA to mlp.fc1 in the last-N blocks")
    parser.add_argument("--ft_lr", type=float, default=DEFAULT_FT_LR)
    parser.add_argument("--ft_epochs", type=int, default=DEFAULT_FT_EPOCHS)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ft_warmup", type=float, default=0.05)
    parser.add_argument("--ft_val_split", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--use_cosface", action="store_true", help="Use CosFace head instead of linear CE")
    # argparse (near other flags)
    parser.add_argument("--pool_mode", choices=["cls_only","mean_no_cls","mean_with_cls"],
                    default="cls_only", help="How to pool transformer outputs")



    # Reporting
    parser.add_argument("--make_report", action="store_true")
    parser.add_argument("--report_dir", type=str, default="./reports")
    parser.add_argument("--fpr_focus", type=float, default=0.1)
    parser.add_argument("--k_grid", type=str, default="8,12,18,24,32")
    parser.add_argument("--dimred", type=str, default="umap", choices=["umap","pca","tsne"])
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap samples for CI (AUC/pAUC)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # 1) load data
    root = Path(args.dataset_root)
    data = load_data(root)

    if args.machine not in data:
        print(f"ERROR: '{args.machine}' key not found. Available keys: {list(data.keys())}")
        sys.exit(1)

    train_pairs = data[args.machine]["train"]
    test_pairs  = data[args.machine]["test"]

    print("Train (normal):", sum(1 for _, w in train_pairs if w is not None))
    print("Test  (normal and anomalous):", sum(1 for _, w in test_pairs if w is not None))

    X_train = [w for _, w in train_pairs if w is not None]
    # keep train paths for potential future diagnostics (domain split rarely used in train)
    train_paths = [p for p, w in train_pairs if w is not None]

    test_pairs = [(p, w) for p, w in test_pairs if w is not None]
    paths_test = np.array([p for p, _ in test_pairs], dtype=object)
    X_test     = [w for _, w in test_pairs]
    y_true     = np.array([infer_label_from_path(p) for p in paths_test], dtype=int)
    domains_te = np.array([infer_domain_from_path(p) for p in paths_test])

    # 2) feature extractor
    # inference extractor
    feat = EATFeatureExtractor(model_id=args.model_id, sr=args.sr,
                           target_length=args.target_length, pool_mode=args.pool_mode)


    # 3) Optional LoRA FT
    pretext_log: Dict[str, object] = {}
    if args.do_lora_ft:
        if not PEFT_AVAILABLE:
            print("[WARN] PEFT not available; skipping LoRA fine-tuning.")
        else:
            wavs_for_ft = X_train
            if len(wavs_for_ft) > args.max_train:
                idx = np.random.default_rng(args.seed).choice(len(wavs_for_ft), size=args.max_train, replace=False)
                wavs_for_ft = [wavs_for_ft[i] for i in idx]
                print(f"[LoRA-FT] Using {len(wavs_for_ft)} randomly selected training clips for pseudo-labeling.")
            else:
                print(f"[LoRA-FT] Using all {len(wavs_for_ft)} training clips for pseudo-labeling.")

            print("[LoRA-FT] Extracting frozen pooled features for KMeans …")
            feats_tr_frozen: List[np.ndarray] = []
            tot = math.ceil(len(wavs_for_ft) / args.batch_size)
            for bi, batch in enumerate(chunked(wavs_for_ft, args.batch_size), 1):
                cls = feat.encode_batch(batch)  # masked mean pooled
                feats_tr_frozen.append(cls)
                if bi % 5 == 0 or bi == tot:
                    print(f"  [pseudo] {bi}/{tot} -> {cls.shape}")
            X_frozen = np.vstack(feats_tr_frozen).astype(np.float32)
            print("[LoRA-FT] Frozen features:", X_frozen.shape)

            # auto K
            if args.auto_hparam:
                K_pretext = auto_choose_k(X_frozen)
                print(f"[LoRA-FT] Auto-chosen pretext K via silhouette: {K_pretext}")
            else:
                K_pretext = max(7, min(64, args.n_clusters))
                print(f"[LoRA-FT] Pretext K set to {K_pretext} (from --n_clusters)")

            scaler_pre = StandardScaler().fit(X_frozen)
            X_frozen_std = scaler_pre.transform(X_frozen)
            km_pre = KMeans(n_clusters=K_pretext, n_init="auto", random_state=args.seed)
            y_pseudo = km_pre.fit_predict(X_frozen_std)

            pretext_log.update({
                "K_pretext": K_pretext,
                "pseudo_label_counts": {int(k): int(v) for k, v in zip(*np.unique(y_pseudo, return_counts=True))},
            })

            N = len(wavs_for_ft)
            idx_all = np.arange(N)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(idx_all)
            n_val = max(1, int(N * args.ft_val_split))
            val_idx = idx_all[:n_val]
            tr_idx  = idx_all[n_val:]

            ds_tr = PseudoLabelDataset([wavs_for_ft[i] for i in tr_idx], y_pseudo[tr_idx], args.sr, args.target_length)
            ds_va = PseudoLabelDataset([wavs_for_ft[i] for i in val_idx], y_pseudo[val_idx], args.sr, args.target_length)
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=collate_fb)
            dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=collate_fb)

            # training extractor (for LoRA FT)
            feat_train = EATFeatureExtractor(model_id=args.model_id, sr=args.sr,
                                            target_length=args.target_length, train_mode=True,
                                            device=feat.device, pool_mode=args.pool_mode)

            # ---- LoRA target resolution ----
            try:
                tokens = resolve_lora_targets(feat_train.model, args.lora_targets)  # e.g., ["qkv","proj"]
            except ValueError as e:
                print("[LoRA-FT][ERROR]", e)
                print("[LoRA-FT] Skipping fine-tuning and continuing with frozen EAT.")
                tokens = []

            target_names = []
            if tokens:
                target_names = find_lora_target_names(
                    feat_train.model, tokens=tokens, last_n=args.lora_last_n,
                    include_only_attn=True, include_mlp_fc1=args.lora_include_mlp_fc1
                )
                if not target_names:
                    print("[LoRA-FT][WARN] No module names matched in last-N blocks; falling back to all blocks for those tokens.")
                    target_names = [n for n, m in feat_train.model.named_modules()
                                    if isinstance(m, nn.Linear)
                                    and any(t in n for t in tokens)
                                    and (".attn." in n or (args.lora_include_mlp_fc1 and n.endswith(".mlp.fc1")))]
            if target_names:
                print(f"[LoRA-FT] Target modules for LoRA (count={len(target_names)}):")
                for n in target_names[:30]:
                    print("   ", n)
                if len(target_names) > 30:
                    print(f"   ... and {len(target_names)-30} more")

                # Auto hyperparams
                r = args.lora_r
                alpha = args.lora_alpha
                dropout = args.lora_dropout
                ft_lr = args.ft_lr
                ft_epochs = args.ft_epochs
                if args.auto_hparam:
                    r = 8 if K_pretext <= 12 else (16 if K_pretext <= 24 else 32)
                    alpha = 2 * r
                    dropout = 0.05 if K_pretext <= 18 else 0.075
                    ft_lr = 1e-4 if device.type == 'cuda' else 5e-5
                    ft_epochs = 5 if N >= 512 else 8
                    print(f"[LoRA-FT] Auto hparams -> r={r}, alpha={alpha}, dropout={dropout}, lr={ft_lr}, epochs={ft_epochs}")

                lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                    target_modules=target_names,
                )
                feat_train.model = get_peft_model(feat_train.model, lora_cfg)

                # infer dim
                with torch.no_grad():
                    sample_fb, _, sample_len = next(iter(dl_tr))
                    sample_fb = sample_fb.to(device)
                    sample_len = sample_len.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type=='cuda'), dtype=amp_dtype):
                        cls_sample = feat_train.forward_cls(sample_fb, lengths=sample_len)
                    in_dim = int(cls_sample.shape[-1])

                # head
                if args.use_cosface:
                    head = CosFaceHead(in_dim, K_pretext).to(device)
                else:
                    head = LinearHead(in_dim, K_pretext).to(device)

                # optimizer & schedule
                optim = torch.optim.AdamW(list(feat_train.model.parameters()) + list(head.parameters()), lr=ft_lr)
                steps_per_epoch = max(1, math.ceil(len(dl_tr)))
                total_steps = steps_per_epoch * ft_epochs
                warmup_steps = int(total_steps * args.ft_warmup)
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

                scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda' and amp_dtype in (torch.float16,)))
                best_val = float('inf')
                bad_epochs = 0

                mu = torch.tensor(scaler_pre.mean_,  dtype=torch.float32, device=device)
                sg = torch.tensor(scaler_pre.scale_, dtype=torch.float32, device=device)

                # Per-epoch diagnostics buffer
                per_epoch_logs = []

                print("[LoRA-FT] Starting training …")
                for epoch in range(1, ft_epochs + 1):
                    feat_train.model.train(); head.train()
                    running = 0.0
                    # simple grad-norm aggregator for LoRA params
                    grad_norm_sum = 0.0; grad_norm_cnt = 0

                    for xb, yb, lengths in dl_tr:
                        xb = xb.to(device); yb = yb.to(device); lengths = lengths.to(device)
                        optim.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast(enabled=(device.type=='cuda'), dtype=amp_dtype):
                            cls = feat_train.forward_cls(xb, lengths=lengths)  # [B,D]
                            cls_std = (cls - mu) / sg
                            logits = head(cls_std, yb) if args.use_cosface else head(cls_std)
                            loss = F.cross_entropy(logits, yb)
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                            if args.grad_clip and args.grad_clip > 0:
                                scaler.unscale_(optim)
                                nn.utils.clip_grad_norm_(list(feat_train.model.parameters()) + list(head.parameters()), args.grad_clip)
                            # grad norm (post unscale)
                            for n, p in feat_train.model.named_parameters():
                                if p.grad is None: continue
                                if "lora_" in n:
                                    g = p.grad.detach()
                                    grad_norm_sum += float(torch.norm(g, p=2).item())
                                    grad_norm_cnt += 1
                            scaler.step(optim)
                            scaler.update()
                        else:
                            loss.backward()
                            if args.grad_clip and args.grad_clip > 0:
                                nn.utils.clip_grad_norm_(list(feat_train.model.parameters()) + list(head.parameters()), args.grad_clip)
                            for n, p in feat_train.model.named_parameters():
                                if p.grad is None: continue
                                if "lora_" in n:
                                    g = p.grad.detach()
                                    grad_norm_sum += float(torch.norm(g, p=2).item())
                                    grad_norm_cnt += 1
                            optim.step()
                        sched.step()
                        running += float(loss.item())
                    train_loss = running / max(1, len(dl_tr))
                    avg_grad_norm = (grad_norm_sum / max(1, grad_norm_cnt))

                    feat_train.model.eval(); head.eval()
                    val_running = 0.0
                    with torch.no_grad():
                        for xb, yb, lengths in dl_va:
                            xb = xb.to(device); yb = yb.to(device); lengths = lengths.to(device)
                            with torch.cuda.amp.autocast(enabled=(device.type=='cuda'), dtype=amp_dtype):
                                cls = feat_train.forward_cls(xb, lengths=lengths)
                                cls_std = (cls - mu) / sg
                                logits = head(cls_std, yb) if args.use_cosface else head(cls_std)
                                loss = F.cross_entropy(logits, yb)
                            val_running += float(loss.item())
                    val_loss = val_running / max(1, len(dl_va))

                    # quick geometry probe on a small batch
                    with torch.no_grad():
                        xb, yb, lengths = next(iter(dl_va))
                        xb = xb.to(device); lengths = lengths.to(device)
                        cls = feat_train.forward_cls(xb, lengths=lengths)
                        cls_std = ((cls - mu) / sg).detach().cpu().numpy()
                        # re-fit tiny KMeans for geometry proxy
                        Ktmp = min( max(4, int(np.sqrt(cls_std.shape[0]))), max(2, K_pretext) )
                        km_tmp = KMeans(n_clusters=Ktmp, n_init="auto", random_state=0).fit(cls_std)
                        labs = km_tmp.labels_
                        # S_W and S_B (proxy)
                        S_W = 0.0
                        Cs = []
                        for k in range(Ktmp):
                            Xk = cls_std[labs==k]
                            if len(Xk) <= 1: continue
                            cov = np.cov(Xk.T)
                            S_W += np.trace(cov)
                            Cs.append(Xk.mean(0))
                        Cs = np.array(Cs) if len(Cs)>0 else np.zeros((Ktmp, cls_std.shape[1]))
                        if len(Cs) >= 2:
                            dmat = np.linalg.norm(Cs[:,None,:]-Cs[None,:,:], axis=2)
                            S_B = float(np.min(dmat[np.triu_indices_from(dmat,1)]))
                        else:
                            S_B = 0.0
                        geom_ratio = float(S_B / max(1e-6, S_W))
                    per_epoch_logs.append({
                        "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "avg_lora_grad_norm": avg_grad_norm, "geom_SB_over_SW": geom_ratio
                    })

                    print(f"[LoRA-FT] epoch {epoch}/{ft_epochs} train={train_loss:.4f} val={val_loss:.4f} | grad||={avg_grad_norm:.3e} | S_B/S_W≈{geom_ratio:.4e}")

                    if val_loss + 1e-6 < best_val:
                        best_val = val_loss
                        bad_epochs = 0
                        best_state = {"model": feat_train.model.state_dict(), "head": head.state_dict()}
                    else:
                        bad_epochs += 1
                        if bad_epochs >= args.early_stop_patience:
                            print("[LoRA-FT] Early stopping.")
                            break

                if 'best_state' in locals():
                    feat_train.model.load_state_dict(best_state["model"])
                    head.load_state_dict(best_state["head"])

                trn, totp = count_trainable_params(feat_train.model)
                print(f"[LoRA-FT] trainable params: {trn:,} | all params: {totp:,} | trainable%: {100.0*trn/totp:.4f}")

                # swap in tuned model for inference
                feat.model = feat_train.model
                
                export_seq_bt128_onnx(feat.model, "./exports/eat_tokens.onnx",
                      example_T=args.target_length, drop_cls=False, match_input_T=False)
                print("[LoRA-FT] ONNX export done.")

                orig_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feat.model.to(orig_device).eval()

                # feat.model.eval()

                pretext_log.update({
                    "lora_tokens": tokens,
                    "lora_target_names_count": len(target_names),
                    "lora_last_n": args.lora_last_n,
                    "lora_include_mlp_fc1": bool(args.lora_include_mlp_fc1),
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "lora_dropout": dropout,
                    "ft_lr": ft_lr,
                    "ft_epochs": ft_epochs,
                    "best_val_loss": best_val,
                    "use_cosface": bool(args.use_cosface),
                    "per_epoch": per_epoch_logs,
                })
            else:
                print("[LoRA-FT] No valid LoRA targets resolved; continuing frozen.")

    # 4) Train final anomaly KMeans
    print(f"Fitting on healthy {args.machine} clips (cap={args.max_train}) …")
    wavs_tr = X_train
    if len(wavs_tr) > args.max_train:
        idx = np.random.default_rng(args.seed).choice(len(wavs_tr), size=args.max_train, replace=False)
        wavs_tr = [wavs_tr[i] for i in idx]
        print(f"Using {len(wavs_tr)} randomly selected training clips.")
    else:
        print(f"Using all {len(wavs_tr)} training clips.")

    feats_tr: List[np.ndarray] = []
    tot = math.ceil(len(wavs_tr) / args.batch_size)
    for bi, batch in enumerate(chunked(wavs_tr, args.batch_size), 1):
        cls = feat.encode_batch(batch)
        feats_tr.append(cls)
        if bi % 10 == 0 or bi == tot:
            print(f"[train] {bi}/{tot} -> {cls.shape}")
    X_tr = np.vstack(feats_tr) if feats_tr else np.zeros((0, 768), dtype=np.float32)
    print("Train features:", X_tr.shape)

    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init="auto", random_state=args.seed)
    kmeans.fit(X_tr_std)
    print("Training completed (KMeans fitted).")

    radii = compute_cluster_radii(X_tr_std, kmeans.labels_, kmeans.cluster_centers_)
    # store for reporting

    # 5) Score test
    print(f"Scoring {len(X_test)} test clips (batch={args.batch_size}) …")
    feats_te: List[np.ndarray] = []
    tot_te = math.ceil(len(X_test) / args.batch_size)
    for bi, batch in enumerate(chunked(X_test, args.batch_size), 1):
        cls = feat.encode_batch(batch)
        feats_te.append(cls)
        if bi % 10 == 0 or bi == tot_te:
            print(f"[test] {bi}/{tot_te} -> {cls.shape}")
    X_te = np.vstack(feats_te) if feats_te else np.zeros((0, X_tr.shape[1]), dtype=np.float32)

    X_te_std = scaler.transform(X_te)
    dmat_te = np.linalg.norm(X_te_std[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2)
    nearest_idx = dmat_te.argmin(axis=1)
    d1 = dmat_te[np.arange(len(X_te_std)), nearest_idx]
    # normalized distance by centroid median radius (avoid div by zero)
    r = radii[nearest_idx]
    scores = (d1 - r) / (r + 1e-6)
    # also store second-best distance for margin analysis
    part_sorted = np.partition(dmat_te, 1, axis=1)
    d1 = part_sorted[:,0]; d2 = part_sorted[:,1]; margin = d2 - d1

    assert len(scores) == len(paths_test)
    print("Scoring completed! First 10 anomaly scores:", np.asarray(scores[:10]).round(4))

    # 6) metrics (existing util + print)
    out_dir = Path(f"./results_{args.machine}"); out_dir.mkdir(parents=True, exist_ok=True)
    domains = [infer_domain_from_path(p) for p in paths_test]
    for dom in ("source", "target"):
        idx = [i for i, d in enumerate(domains) if d == dom]
        if not idx:
            print(f"No '{dom}' samples detected in test set.")
            continue
        dom_paths  = [paths_test[i] for i in idx]
        dom_scores = scores[idx]
        dom_out    = out_dir / dom
        dom_metrics = compute_and_save_metrics(paths=dom_paths, scores=dom_scores, out_dir=dom_out)
        print(f"\n[{dom}] AUC={dom_metrics['auc']:.4f}, pAUC={dom_metrics['pauc']:.4f} (saved to {dom_out})")

    metrics_all = compute_and_save_metrics(paths=paths_test, scores=scores, out_dir=out_dir)
    print(f"\n[all] AUC  = {metrics_all['auc']:.4f}")
    print(f"[all] pAUC = {metrics_all['pauc']:.4f} (saved to {out_dir})")

    # ---------------- REPORT GENERATION ----------------
    if args.make_report:
        run_tag = ("lora" if args.do_lora_ft else "baseline")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        report_root = Path(args.report_dir) / args.machine / f"{run_tag}_{stamp}"
        report_root.mkdir(parents=True, exist_ok=True)
        figs = report_root / "figs"; figs.mkdir(exist_ok=True)

        # Save numpy artifacts for cross-run comparison
        np.save(report_root / "X_tr_std.npy", X_tr_std)
        np.save(report_root / "X_te_std.npy", X_te_std)
        np.save(report_root / "centers.npy", kmeans.cluster_centers_)
        np.save(report_root / "scores.npy", scores)
        np.save(report_root / "nearest_idx.npy", nearest_idx)
        np.save(report_root / "d1.npy", d1); np.save(report_root / "d2.npy", d2); np.save(report_root / "margin.npy", margin)
        with open(report_root / "paths_test.txt", "w", encoding="utf-8") as f:
            for p in paths_test: f.write(str(p)+"\n")
        np.save(report_root / "y_true.npy", y_true)
        np.save(report_root / "domains_te.npy", domains_te)

        # 1) K selection on train
        K_list = [int(k) for k in args.k_grid.split(",") if k.strip()]
        k_panel_rows = []
        for K in K_list:
            km = KMeans(n_clusters=K, n_init="auto", random_state=args.seed).fit(X_tr_std)
            labs = km.labels_
            sil = silhouette_score(X_tr_std, labs) if len(np.unique(labs))>1 else -1
            db  = davies_bouldin_score(X_tr_std, labs)
            ch  = calinski_harabasz_score(X_tr_std, labs)
            k_panel_rows.append({"K": K, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "inertia": float(km.inertia_)})

        # 2) Silhouette plot for current K
        sil_mean = silhouette_plot(X_tr_std, kmeans.labels_, title="Silhouette (train normals)", path=str(figs/"silhouette_train.png"))

        # 3) Centroid heatmap + within/ between stats
        centroid_distance_heatmap(kmeans.cluster_centers_, "Centroid pairwise distances", str(figs/"centroid_dist_heatmap.png"))
        within_rows = []
        for cid in range(args.n_clusters):
            Xc = X_tr_std[kmeans.labels_==cid]
            if len(Xc) >= 2:
                cov = np.cov(Xc.T)
                wtrace = float(np.trace(cov))
            else:
                wtrace = 0.0
            within_rows.append({"cluster": cid, "size": int(len(Xc)), "within_trace": wtrace})
        # between (min distance between centroids)
        C = kmeans.cluster_centers_
        dC = np.linalg.norm(C[:,None,:]-C[None,:,:], axis=2)
        mask = np.triu(np.ones_like(dC, dtype=bool), k=1)
        min_between = float(dC[mask].min()) if mask.sum() else 0.0
        SB_over_SW = float(min_between / max(1e-6, sum(r["within_trace"] for r in within_rows)))

        # 4) 2D projections
        Ztr = make_2d(X_tr_std, method=args.dimred, seed=args.seed)
        Zte = make_2d(X_te_std, method=args.dimred, seed=args.seed)
        ZC  = make_2d(kmeans.cluster_centers_, method="pca", seed=args.seed)  # stable for centers

        plot_2d(Ztr, labels=kmeans.labels_, title="Train (normals) clusters", path=str(figs/"train_2d.png"), centroids=None)
        plot_2d(Zte, labels=y_true, title="Test (0=normal,1=anomaly)", path=str(figs/"test_2d.png"), centroids=None)
        # Overlay centers onto test projection (align spaces via same reducer is better; approximating with PCA for centers)
        plot_2d(Zte, labels=y_true, title="Test with centroids (approx)", path=str(figs/"test_with_centroids.png"),
                centroids=ZC)

        # 5) Score distributions
        def plot_hist_cdf(scores, y_true, title, base):
            # Histogram
            plt.figure(figsize=(6,4))
            plt.hist(scores[y_true==0], bins=50, alpha=0.6, density=True, label="normal")
            plt.hist(scores[y_true==1], bins=50, alpha=0.6, density=True, label="anomaly")
            plt.title(title+" (hist)"); plt.legend(); plt.tight_layout()
            plt.savefig(figs/f"{base}_hist.png", dpi=170); plt.close()
            # CDF
            def ecdf(x): x=np.sort(x); y=np.linspace(0,1,len(x)); return x,y
            nX, nY = ecdf(scores[y_true==0]); aX, aY = ecdf(scores[y_true==1])
            plt.figure(figsize=(6,4))
            plt.plot(nX, nY, label="normal")
            plt.plot(aX, aY, label="anomaly")
            plt.title(title+f" (CDF)"); plt.xlabel("score"); plt.ylabel("CDF"); plt.legend(); plt.tight_layout()
            plt.savefig(figs/f"{base}_cdf.png", dpi=170); plt.close()
        plot_hist_cdf(scores, y_true, "Score distributions (all)", "score_all")

        # Domain split scores (if both present)
        have_src = np.any(domains_te=="source"); have_tgt = np.any(domains_te=="target")
        if have_src:
            plot_hist_cdf(scores[domains_te=="source"], y_true[domains_te=="source"], "Scores (source)", "score_source")
        if have_tgt:
            plot_hist_cdf(scores[domains_te=="target"], y_true[domains_te=="target"], "Scores (target)", "score_target")

        # 6) ROC & pAUC
        A_all, pA_all = plot_roc(y_true, scores, f"ROC (all) — focus {args.fpr_focus}", str(figs/"roc_all.png"), fpr_max=args.fpr_focus)
        A_src=pA_src=A_tgt=pA_tgt=None
        if have_src:
            A_src, pA_src = plot_roc(y_true[domains_te=="source"], scores[domains_te=="source"], f"ROC (source)", str(figs/"roc_source.png"), fpr_max=args.fpr_focus)
        if have_tgt:
            A_tgt, pA_tgt = plot_roc(y_true[domains_te=="target"], scores[domains_te=="target"], f"ROC (target)", str(figs/"roc_target.png"), fpr_max=args.fpr_focus)

        # CIs (bootstrap)
        ci_auc = bootstrap_ci(y_true, scores, lambda yt,sc: auc(*roc_curve(yt,sc)[:2]), n=args.bootstrap, seed=args.seed)
        ci_pauc = bootstrap_ci(y_true, scores, lambda yt,sc: pauc_at_fpr(yt,sc,args.fpr_focus)[0], n=args.bootstrap, seed=args.seed)

        # 7) Threshold table
        thr_table = threshold_table(y_true, scores, fprs=(0.005,0.01,0.02,0.05,0.1))
        # choose operating threshold at closest to target fpr_focus
        target_thr = [t for t in thr_table if abs(t["target_fpr"]-args.fpr_focus) < 1e-9]
        op_thr = target_thr[0]["threshold"] if target_thr else np.percentile(scores, 95)

        # 8) Error analysis at operating point
        y_pred = (scores >= op_thr).astype(int)
        err_fp_idx = np.where((y_true==0) & (y_pred==1))[0]
        err_fn_idx = np.where((y_true==1) & (y_pred==0))[0]
        # rank by margin (smaller margin = more ambiguous)
        fp_order = np.argsort(margin[err_fp_idx])  # low margin worst
        fn_order = np.argsort(margin[err_fn_idx])
        err_fp = [{
            "path": str(paths_test[i]), "domain": str(domains_te[i]), "score": float(scores[i]),
            "nearest_centroid": int(nearest_idx[i]), "d1": float(d1[i]), "d2": float(d2[i]), "margin": float(margin[i])
        } for i in err_fp_idx[fp_order]]
        err_fn = [{
            "path": str(paths_test[i]), "domain": str(domains_te[i]), "score": float(scores[i]),
            "nearest_centroid": int(nearest_idx[i]), "d1": float(d1[i]), "d2": float(d2[i]), "margin": float(margin[i])
        } for i in err_fn_idx[fn_order]]

        # 9) Domain shift metrics (on normals only, to avoid contamination)
        shift_rows = []
        if have_src and have_tgt:
            norm_src = (y_true==0) & (domains_te=="source")
            norm_tgt = (y_true==0) & (domains_te=="target")
            if np.any(norm_src) and np.any(norm_tgt):
                Xs = X_te_std[norm_src]; Xt = X_te_std[norm_tgt]
                mu_s, C_s = gaussian_stats(Xs); mu_t, C_t = gaussian_stats(Xt)
                fid_st = frechet_distance(mu_s, C_s, mu_t, C_t)
                mmd_st = mmd_rbf(Xs, Xt)
                shift_rows.append({"pair":"source_vs_target (normals)", "frechet": fid_st, "mmd_rbf": mmd_st})

        # 10) Excel report
        summary = {
            "machine": args.machine,
            "run": run_tag,
            "AUC_all": A_all, "pAUC_all": pA_all,
            "AUC_source": A_src, "pAUC_source": pA_src,
            "AUC_target": A_tgt, "pAUC_target": pA_tgt,
            "AUC_all_CI95": ci_auc, "pAUC_all_CI95": ci_pauc,
            "K": args.n_clusters,
            "silhouette_train": sil_mean,
            "SB_over_SW_train": SB_over_SW,
            "op_threshold@fpr_focus": float(op_thr),
            "do_lora_ft": bool(args.do_lora_ft),
            "lora_last_n": int(args.lora_last_n),
            "lora_targets": pretext_log.get("lora_tokens"),
            "lora_r": pretext_log.get("lora_r"),
            "lora_alpha": pretext_log.get("lora_alpha"),
            "lora_dropout": pretext_log.get("lora_dropout"),
            "use_cosface": pretext_log.get("use_cosface"),
            "ft_epochs": pretext_log.get("ft_epochs"),
            "best_val_loss": pretext_log.get("best_val_loss"),
        }

        train_clusters = []
        for r in within_rows:
            cid = r["cluster"]
            mu = kmeans.cluster_centers_[cid]
            train_clusters.append({
                "cluster": cid, "size": r["size"], "within_trace": r["within_trace"],
                **{f"mu_{j}": float(mu[j]) for j in range(min(8, mu.shape[0]))}  # first 8 dims preview
            })

        test_scores_tbl = []
        for i in range(len(paths_test)):
            test_scores_tbl.append({
                "path": str(paths_test[i]), "domain": str(domains_te[i]), "label": int(y_true[i]),
                "score": float(scores[i]), "nearest_centroid": int(nearest_idx[i]),
                "d1": float(d1[i]), "d2": float(d2[i]), "margin": float(margin[i]),
            })

        k_panel_df = k_panel_rows
        thr_df = thr_table

        per_epoch = pretext_log.get("per_epoch", [])

        # Write Excel
        xlsx_path = report_root / "report.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
            pd.DataFrame([summary]).to_excel(xl, sheet_name="Summary", index=False)
            pd.DataFrame(train_clusters).to_excel(xl, sheet_name="Train_Clusters", index=False)
            pd.DataFrame(test_scores_tbl).to_excel(xl, sheet_name="Test_Scores", index=False)
            pd.DataFrame(err_fp).to_excel(xl, sheet_name="Errors_FP", index=False)
            pd.DataFrame(err_fn).to_excel(xl, sheet_name="Errors_FN", index=False)
            pd.DataFrame(k_panel_df).to_excel(xl, sheet_name="K_Selection", index=False)
            pd.DataFrame(thr_df).to_excel(xl, sheet_name="Thresholds", index=False)
            if shift_rows:
                pd.DataFrame(shift_rows).to_excel(xl, sheet_name="Domain_Shift", index=False)
            if per_epoch:
                pd.DataFrame(per_epoch).to_excel(xl, sheet_name="Per_Epoch", index=False)

        # Save a README.txt with figure list
        with open(report_root / "README.txt", "w", encoding="utf-8") as f:
            f.write("Generated diagnostics:\n")
            for p in sorted(figs.glob("*.png")):
                f.write(f"- {p.name}\n")
            f.write("\nSee report.xlsx for tables.\n")

        print(f"\n[REPORT] Comprehensive report saved to: {report_root}")
        print(f"         Excel: {xlsx_path}")
        print(f"         Figures: {figs}")

    # Manifest
    out_dir = Path(f"./results_{args.machine}")
    manifest = {
        "model_id": args.model_id,
        "sr": args.sr,
        "target_length": args.target_length,
        "n_clusters": args.n_clusters,
        "batch_size": args.batch_size,
        "max_train": args.max_train,
        "train_feat_shape": list(X_tr.shape),
        "test_feat_shape": list(X_te.shape),
        "do_lora_ft": bool(args.do_lora_ft),
        "pretext": {k: (v if k!="per_epoch" else f"{len(v)} entries") for k,v in (pretext_log or {}).items()},
        "device": str(device),
        "amp_dtype": str(amp_dtype),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
