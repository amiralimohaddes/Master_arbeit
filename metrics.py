# metrics.py
"""
Utility to compute and persist evaluation metrics for the fan-anomaly task.

Public API
----------
compute_and_save_metrics(paths, scores, out_dir, max_fpr=0.1)  →  dict

    • paths   : list[str]  absolute or relative file names (same order as scores)
    • scores  : 1-D array-like of anomaly scores
    • out_dir : pathlib.Path | str – folder will be created if it does not exist
    • max_fpr : float       – upper bound for the partial AUC (default 0.10)

Saves
-----
out_dir/
    ├─ scores.csv   (path,score,label)
    ├─ metrics.txt  (AUC, pAUC)
    └─ roc.png      (full ROC curve, dotted line at `max_fpr`)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


# --------------------------------------------------------------------------
def _label_from_path(p: str | Path) -> int:
    """0 = normal, 1 = anomaly   (rule from `load_dataset.py`)."""
    stem = Path(p).stem.lower()
    return 0 if "normal" in stem else 1


def _partial_auc(fpr: np.ndarray, tpr: np.ndarray, *, max_fpr: float) -> float:
    """Trapezoidal AUC between FPR=0 and `max_fpr`, then normalised to [0,1]."""
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) < 2:                       # not enough points in that region
        return float("nan")
    return auc(fpr[idx], tpr[idx]) / max_fpr


# --------------------------------------------------------------------------
def compute_and_save_metrics(
    paths: Sequence[str | Path],
    scores: Sequence[float],
    out_dir: str | Path,
    *,
    max_fpr: float = 0.10,
) -> dict:
    """Compute AUC / pAUC and persist artefacts.  Returns a metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list(map(str, paths))
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.fromiter((_label_from_path(p) for p in paths), dtype=np.int8)

    # full-range ROC & AUC
    auc_full = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # partial AUC up to `max_fpr`
    pauc = _partial_auc(fpr, tpr, max_fpr=max_fpr)

    # ── save artefacts ───────────────────────────────────────────────────
    # 1. CSV with individual scores
    with open(out_dir / "scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score", "label"])
        writer.writerows(zip(paths, scores, labels))

    # 2. plain-text summary
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"AUC        : {auc_full:.6f}\n")
        f.write(f"pAUC@{max_fpr:.2f}: {pauc:.6f}\n")

    # 3. ROC curve figure
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_full:.3f}")
    plt.axvline(max_fpr, ls="--", lw=0.8, color="grey", label=f"{max_fpr:.0%} FPR")
    plt.title("ROC curve – fan")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=160)
    plt.close()

    return {"auc": auc_full, "pauc": pauc}


# --------------------------------------------------------------------------
if __name__ == "__main__":  # standalone CLI (optional)
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("scores_csv", type=Path, help="CSV with path,score,label")
    ap.add_argument("--out", type=Path, default=Path("./results"), help="output dir")
    ap.add_argument("--max_fpr", type=float, default=0.10, help="FPR upper bound")
    args = ap.parse_args()

    rows = list(csv.reader(open(args.scores_csv, newline="")))
    header = rows.pop(0)
    path_idx = header.index("path")
    score_idx = header.index("score")
    label_idx = header.index("label")

    paths  = [r[path_idx]   for r in rows]
    scores = [float(r[score_idx]) for r in rows]
    labels = [int(r[label_idx])   for r in rows]

    # call metric routine with explicit labels (ignore auto-labelling)
    metrics = compute_and_save_metrics(paths, scores, args.out, max_fpr=args.max_fpr)
    # overwrite with ground-truth p/r to be 100 % accurate
    metrics["auc"]  = roc_auc_score(labels, scores)
    fpr, tpr, _     = roc_curve(labels, scores)
    metrics["pauc"] = _partial_auc(fpr, tpr, max_fpr=args.max_fpr)

    # pretty print
    print(json.dumps(metrics, indent=2))
