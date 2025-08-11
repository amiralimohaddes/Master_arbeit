#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 fileformat=unix :

from pathlib import Path
from load_dataset import load_data
import sys
import numpy as np

# ready-made helper
from to_install.unsupervised_anomalie_detection.sklearn_pipes import get_deep_feat_cls_eat_all_model




# 1) load data -------------------------------------------------------------
root      = Path("./dataset")
data      = load_data(root)

print("Train (normal):", len(data["bearing"]["train"]))
print("Test  (normal and anomalous):", len(data["bearing"]["test"]))

X_train   = [wav for _, wav in data["bearing"]["train"] if wav is not None]


# 2) build pipeline --------------------------------------------------------
pipe, _   = get_deep_feat_cls_eat_all_model(
                feat_model_type   = "EAT",   # <- real Efficient Audio Transformer
                feat_model_depth  = 12,      # matches your checkpoint depth
                n_components      = 10,       # K-means clusters / PCA dims / GMM comps
                model_type        = "kmeansNN"   # or 'pca_recon' / 'GMMpb'
            )



# 3) train & score ---------------------------------------------------------
# Batch processing configuration
BATCH_SIZE = 32  # Adjust this based on your memory capacity

print(f"Fitting on {len(X_train)} healthy bearing clips...")

# Instead of trying to use partial_fit, we'll fit on a subset of the training data
# This is a trade-off between memory usage and model quality
MAX_TRAINING_SAMPLES = 400  # Adjust based on your memory capacity and needed accuracy

if len(X_train) > MAX_TRAINING_SAMPLES:
    print(f"Using {MAX_TRAINING_SAMPLES} random samples out of {len(X_train)} for training")
    # Use random selection for better representation
    np.random.seed(42)  # For reproducibility
    train_indices = np.random.choice(len(X_train), MAX_TRAINING_SAMPLES, replace=False)
    train_subset = [X_train[i] for i in train_indices]
    pipe.fit(train_subset)
else:
    print(f"Using all {len(X_train)} samples for training")
    pipe.fit(X_train)

print("Training completed!")

# ----- 1. keep only usable clips *and* remember their paths -------------
test_pairs = [(p, wav) for p, wav in data["bearing"]["test"] if wav is not None]

paths_test = [p   for p, _ in test_pairs]   # length N
X_test     = [wav for _, wav in test_pairs] # length N

# sanity-check
assert len(paths_test) == len(X_test), "path/clip count mismatch"

print(f"Generating anomaly scores for {len(X_test)} test clips in batches of {BATCH_SIZE}...")

# Process test data in batches
all_scores = []
total_test_batches = (len(X_test) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(X_test), BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, len(X_test))
    batch = X_test[i:batch_end]
    current_batch = (i // BATCH_SIZE) + 1
    
    print(f"Processing test batch {current_batch}/{total_test_batches} ({len(batch)} samples)...", flush=True)
    
    # Transform in smaller sub-batches if needed to further reduce memory usage
    SUB_BATCH_SIZE = 8  # Even smaller batches for transform if needed
    batch_scores = []
    
    for j in range(0, len(batch), SUB_BATCH_SIZE):
        sub_end = min(j + SUB_BATCH_SIZE, len(batch))
        sub_batch = batch[j:sub_end]
        sub_scores = pipe.transform(sub_batch).ravel()
        batch_scores.extend(sub_scores)
    
    all_scores.extend(batch_scores)

scores = np.array(all_scores)

assert len(scores) == len(paths_test)
print("Scoring completed!")

print("Anomaly scores:\n", scores)

from metrics import compute_and_save_metrics
out_dir = Path("./results_bearing")           # or any folder you like
metrics = compute_and_save_metrics(
    paths=paths_test,
    scores=scores,
    out_dir=out_dir,
)

print(f"\nAUC  = {metrics['auc']:.4f}")
print(f"pAUC = {metrics['pauc']:.4f} (saved to {out_dir})")


