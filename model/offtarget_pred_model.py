#!/usr/bin/env python3
"""
Enhanced Off-Target Prediction Model with Heterogeneous Graph Learning
======================================================================

Improvements over offtarget_pred_model_05_03.py:

Phase 2 - Data splitting fixes:
  - F2: Separate MP and supervision edges for training (90/10 internal split)
  - F6: Configurable neg:pos ratio (current best 3:1)
  - F7: Inductive guide-level split option (hold out entire guides for test)
  - Improved hard negative sampling with seed-region awareness

Phase 3 - Architecture overhaul:
  - F3:  TransformerConv replaces SAGEConv (natively supports edge features via edge_dim)
  - F11: Reverse edges carry mirrored edge features for bidirectional MP
  - F12: Edge feature encoder with BatchNorm before decoder
  - F13: Projection-based skip connections (no silent disabling)

Phase 4 - Training refinements:
  - F8:  Removed `abc` breakpoint
  - F9:  Checkpoint saves/loads full hyperparameters for consistent reconstruction
  - F10: FocalLoss replaces label-smoothed BCE + pos_weight

Phase 5 - Evaluation & interpretability:
  - Per-guide AUPRC breakdown
  - Fixed calibrated operating threshold for binary decisions
    - Full-model train/validation/test diagnostics
"""

import os
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, TransformerConv, Linear
from torch_geometric.data import HeteroData

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, precision_score, recall_score
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (repository-relative by default; override via env vars if needed)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "model"

GRAPH_DATA_PATH = Path(
    os.getenv("OFFTARGET_GRAPH_DATA_PATH", str(MODEL_DIR / "hetero_graph_data_new1.pt"))
)
CSV_DATA_PATH = Path(
    os.getenv("OFFTARGET_FILTERED_CSV_PATH", str(DATA_DIR / "offtarget_filtered.csv"))
)
CHECKPOINT_DIR = Path(
    os.getenv("OFFTARGET_CHECKPOINT_DIR", str(MODEL_DIR / "checkpoints"))
)
FIG_DIR = Path(
    os.getenv("OFFTARGET_FIG_DIR", str(REPO_ROOT / "figs"))
)

# Split configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
HARD_NEG_RATIO = 0.5
MIN_MATCH_RATIO = 0.4
MAX_MATCH_RATIO = 0.9
NEG_POS_RATIO = int(os.getenv("OFFTARGET_NEG_POS_RATIO", "3"))
MP_SUPERVISION_RATIO = float(os.getenv("OFFTARGET_MP_SUP_RATIO", "0.9"))
INDUCTIVE_SPLIT = os.getenv("OFFTARGET_INDUCTIVE_SPLIT", "true").lower() in {"1", "true", "yes"}
INDUCTIVE_GUIDE_FRAC = float(os.getenv("OFFTARGET_INDUCTIVE_GUIDE_FRAC", "0.10"))
SEED = int(os.getenv("OFFTARGET_SEED", "42"))

# Model hyperparameters
HIDDEN_CHANNELS = int(os.getenv("OFFTARGET_HIDDEN_CHANNELS", "96"))
OUT_CHANNELS = int(os.getenv("OFFTARGET_OUT_CHANNELS", "96"))
EDGE_FEAT_DIM = int(os.getenv("OFFTARGET_EDGE_FEAT_DIM", "25"))
NUM_LAYERS = int(os.getenv("OFFTARGET_NUM_LAYERS", "3"))
NUM_HEADS = int(os.getenv("OFFTARGET_NUM_HEADS", "4"))
DROPOUT = float(os.getenv("OFFTARGET_DROPOUT", "0.4"))
EDGE_DROPOUT = float(os.getenv("OFFTARGET_EDGE_DROPOUT", "0.3"))
CALIBRATED_THRESHOLD = float(os.getenv("OFFTARGET_CALIBRATED_THRESHOLD", "0.0779"))

# Training hyperparameters
LEARNING_RATE = float(os.getenv("OFFTARGET_LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.getenv("OFFTARGET_WEIGHT_DECAY", "5e-3"))
NUM_EPOCHS = int(os.getenv("OFFTARGET_NUM_EPOCHS", "500"))
PATIENCE = int(os.getenv("OFFTARGET_PATIENCE", "20"))
GRAD_CLIP = float(os.getenv("OFFTARGET_GRAD_CLIP", "1.0"))

# Device
cuda_device = os.getenv("CUDA_DEVICE", "cuda")
if torch.cuda.is_available():
    try:
        device = torch.device(cuda_device)
    except Exception:
        device = torch.device("cuda")
else:
    device = torch.device('cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if not GRAPH_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Graph file not found: {GRAPH_DATA_PATH}\n"
        "Run `python model/graph_building.py` first to build the heterogeneous graph."
    )
if not CSV_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Filtered CSV not found: {CSV_DATA_PATH}\n"
        "Run `python model/graph_building.py` first to generate data/offtarget_filtered.csv."
    )

print(f"Using graph data: {GRAPH_DATA_PATH}")
print(f"Using filtered CSV: {CSV_DATA_PATH}")
print(f"Checkpoints -> {CHECKPOINT_DIR}")
print(f"Figures -> {FIG_DIR}")
print(f"Device -> {device}")

data_dict = torch.load(GRAPH_DATA_PATH, weights_only=False)
full_graph = data_dict['full_graph']
guide_to_idx = data_dict['guide_to_idx']
site_to_idx = data_dict['site_to_idx']
metadata_info = data_dict['metadata']

print(f"✓ Loaded graph data successfully!")
print(full_graph)

df1 = pd.read_csv(CSV_DATA_PATH)
print(f"CSV shape: {df1.shape}")

# Create sequence mappings
region_to_seq = dict(zip(df1['Target_region'], df1['Target_sequence']))
site_to_seq = {}
missing_count = 0
for region_key, site_idx in site_to_idx.items():
    if region_key in region_to_seq:
        site_to_seq[site_idx] = region_to_seq[region_key]
    else:
        missing_count += 1
        site_to_seq[site_idx] = 'N' * 23

guide_to_seq = {idx: seq for seq, idx in guide_to_idx.items()}

print(f"✓ Sequence mappings: {len(guide_to_seq)} guides, {len(site_to_seq):,} sites")
if missing_count > 0:
    print(f"  ⚠️ {missing_count} sites missing sequences (using placeholder)")


# ============================================================================
# FEATURE COMPUTATION FUNCTIONS (unchanged from original)
# ============================================================================

def compute_gc_content(seq):
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0.0


def align_sequences(guide_seq, target_seq):
    from Bio import pairwise2
    alignments = pairwise2.align.globalxx(guide_seq, target_seq)
    if alignments:
        max_score = alignments[0].score
        max_possible_score = min(len(guide_seq), len(target_seq))
        return max_score / max_possible_score if max_possible_score > 0 else 0.0
    return 0.0


def compute_mismatch_vector(guide_seq, target_seq):
    max_len = max(len(guide_seq), len(target_seq))
    guide_padded = guide_seq.ljust(max_len, 'N')
    target_padded = target_seq.ljust(max_len, 'N')
    match_vector = []
    mismatch_count = 0
    for i in range(max_len):
        if guide_padded[i] == target_padded[i] and guide_padded[i] != 'N':
            match_vector.append(1)
        else:
            match_vector.append(0)
            mismatch_count += 1
    return match_vector, mismatch_count


def compute_weighted_mismatch_score(guide_seq, target_seq, pam_position='end'):
    match_vector, _ = compute_mismatch_vector(guide_seq, target_seq)
    length = len(match_vector)
    if pam_position == 'end':
        weights = np.linspace(0.5, 2.0, length)
    else:
        weights = np.ones(length)
    weighted_mismatch = sum((1 - m) * w for m, w in zip(match_vector, weights))
    max_weighted = sum(weights)
    return weighted_mismatch / max_weighted if max_weighted > 0 else 0.0


def compute_cfd_like_score(guide_seq, target_seq):
    match_vector, _ = compute_mismatch_vector(guide_seq, target_seq)
    length = len(match_vector)
    position_weights = np.linspace(0.5, 1.5, length)
    match_score = sum(m * w for m, w in zip(match_vector, position_weights))
    max_score = sum(position_weights)
    return match_score / max_score if max_score > 0 else 0.0


def compute_melting_temperature(guide_seq, target_seq):
    seq = guide_seq if len(guide_seq) <= len(target_seq) else target_seq
    seq_upper = seq.upper()
    tm = 2 * (seq_upper.count('A') + seq_upper.count('T')) + \
         4 * (seq_upper.count('G') + seq_upper.count('C'))
    return tm


def compute_edge_features_for_edges(edge_label_index, guide_to_seq, site_to_seq,
                                     guide_to_idx, site_to_idx, device):
    guide_indices = edge_label_index[0].cpu().numpy()
    site_indices = edge_label_index[1].cpu().numpy()
    edge_features_list = []
    for g_idx, s_idx in tqdm(zip(guide_indices, site_indices),
                              total=len(guide_indices),
                              desc="Computing edge features", leave=False):
        guide_seq = guide_to_seq[g_idx]
        target_seq = site_to_seq[s_idx]
        match_vector, total_mismatches = compute_mismatch_vector(guide_seq, target_seq)
        if len(match_vector) < 20:
            match_vector = match_vector + [0] * (20 - len(match_vector))
        else:
            match_vector = match_vector[:20]
        mismatch_count_norm = total_mismatches / max(len(guide_seq), len(target_seq)) \
                              if max(len(guide_seq), len(target_seq)) > 0 else 0
        alignment_score = align_sequences(guide_seq, target_seq)
        weighted_mismatch = compute_weighted_mismatch_score(guide_seq, target_seq)
        cfd_score = compute_cfd_like_score(guide_seq, target_seq)
        tm = compute_melting_temperature(guide_seq, target_seq)
        tm_normalized = tm / 100.0
        edge_feature = match_vector + [
            mismatch_count_norm, alignment_score,
            weighted_mismatch, cfd_score, tm_normalized
        ]
        edge_features_list.append(edge_feature)
    return torch.tensor(edge_features_list, dtype=torch.float, device=device)


print("✓ Feature computation functions defined")


# ============================================================================
# PHASE 2: CORRECTED DATA SPLITTING
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 2: CORRECTED DATA SPLITTING")
print("=" * 70)

edge_type = ('guide', 'targets', 'site')
rev_edge_type = ('site', 'rev_targets', 'guide')
all_edges_original = full_graph[edge_type].edge_index
all_edge_features_original = full_graph[edge_type].edge_attr
num_guides = full_graph['guide'].num_nodes
num_sites = full_graph['site'].num_nodes


def compute_match_ratio(guide_seq, site_seq):
    min_len = min(len(guide_seq), len(site_seq))
    if min_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len)
                  if guide_seq[i].upper() == site_seq[i].upper())
    return matches / min_len


def edge_index_to_set(edge_index):
    return set((int(edge_index[0, i]), int(edge_index[1, i]))
               for i in range(edge_index.shape[1]))


def set_to_edge_index(edge_set):
    if len(edge_set) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edges = list(edge_set)
    return torch.tensor(edges, dtype=torch.long).T


# --- Step 1: Deduplicate edges ---
unique_edges_dict = {}
for i in range(all_edges_original.shape[1]):
    edge = (int(all_edges_original[0, i]), int(all_edges_original[1, i]))
    if edge not in unique_edges_dict:
        unique_edges_dict[edge] = i

unique_edges_list = list(unique_edges_dict.keys())
unique_edge_indices = list(unique_edges_dict.values())
all_unique_edges = torch.tensor(unique_edges_list, dtype=torch.long).T
all_unique_features = all_edge_features_original[unique_edge_indices]
num_unique_edges = len(unique_edges_list)
all_pos_set = set(unique_edges_list)

print(f"  Total edges: {all_edges_original.shape[1]:,}")
print(f"  Unique edges: {num_unique_edges:,}")
print(f"  Guides: {num_guides}, Sites: {num_sites:,}")

# --- Step 2: Split edges ---
if INDUCTIVE_SPLIT:
    # Phase 2 F7: Hold out entire guides for inductive evaluation
    print(f"\n✂️ INDUCTIVE SPLIT: Holding out {INDUCTIVE_GUIDE_FRAC:.0%} of guides for test")

    all_guide_ids = list(range(num_guides))
    np.random.shuffle(all_guide_ids)
    n_test_guides = max(1, int(INDUCTIVE_GUIDE_FRAC * num_guides))
    n_val_guides = max(1, int(INDUCTIVE_GUIDE_FRAC * num_guides))
    test_guide_set = set(all_guide_ids[:n_test_guides])
    val_guide_set = set(all_guide_ids[n_test_guides:n_test_guides + n_val_guides])
    train_guide_set = set(all_guide_ids[n_test_guides + n_val_guides:])

    print(f"  Train guides: {len(train_guide_set)}, Val guides: {len(val_guide_set)}, "
          f"Test guides: {len(test_guide_set)}")

    train_pos_list, val_pos_list, test_pos_list = [], [], []
    train_feat_list, val_feat_list, test_feat_list = [], [], []

    for i in range(num_unique_edges):
        g = int(all_unique_edges[0, i])
        if g in test_guide_set:
            test_pos_list.append(i)
        elif g in val_guide_set:
            val_pos_list.append(i)
        else:
            train_pos_list.append(i)

    train_idx = torch.tensor(train_pos_list, dtype=torch.long)
    val_idx = torch.tensor(val_pos_list, dtype=torch.long)
    test_idx = torch.tensor(test_pos_list, dtype=torch.long)
else:
    # Transductive random edge split (original behavior)
    print(f"\n✂️ TRANSDUCTIVE SPLIT: Random {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{1-TRAIN_RATIO-VAL_RATIO:.0%}")
    perm = torch.randperm(num_unique_edges)
    train_size = int(TRAIN_RATIO * num_unique_edges)
    val_size = int(VAL_RATIO * num_unique_edges)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

train_pos_edges = all_unique_edges[:, train_idx]
val_pos_edges = all_unique_edges[:, val_idx]
test_pos_edges = all_unique_edges[:, test_idx]
train_pos_features = all_unique_features[train_idx]
val_pos_features = all_unique_features[val_idx]
test_pos_features = all_unique_features[test_idx]

train_pos_set = edge_index_to_set(train_pos_edges)
val_pos_set = edge_index_to_set(val_pos_edges)
test_pos_set = edge_index_to_set(test_pos_edges)

print(f"  Train pos: {len(train_pos_set):,}, Val pos: {len(val_pos_set):,}, "
      f"Test pos: {len(test_pos_set):,}")

# Verify zero overlap
assert len(train_pos_set & val_pos_set) == 0, "Train-Val positive overlap!"
assert len(train_pos_set & test_pos_set) == 0, "Train-Test positive overlap!"
assert len(val_pos_set & test_pos_set) == 0, "Val-Test positive overlap!"
print("  ✅ Zero positive edge overlap across splits")

# --- Phase 2 F2: Separate MP and supervision within training ---
n_train_pos = train_pos_edges.shape[1]
mp_size = int(MP_SUPERVISION_RATIO * n_train_pos)
train_perm = torch.randperm(n_train_pos)

mp_idx = train_perm[:mp_size]
train_sup_idx = train_perm[mp_size:]

mp_edges = train_pos_edges[:, mp_idx]
mp_features = train_pos_features[mp_idx]
train_sup_pos_edges = train_pos_edges[:, train_sup_idx]
train_sup_pos_features = train_pos_features[train_sup_idx]

mp_set = edge_index_to_set(mp_edges)
train_sup_set = edge_index_to_set(train_sup_pos_edges)
assert len(mp_set & train_sup_set) == 0, "MP-Supervision overlap in training!"

print(f"\n  Phase 2 F2: Train MP edges: {mp_edges.shape[1]:,}, "
      f"Train supervision pos: {train_sup_pos_edges.shape[1]:,}")
print(f"  ✅ MP and supervision edges are disjoint within training split")


# --- Step 3: Negative sampling with configurable ratio ---
all_sampled_negs = set()


def sample_negatives_for_split(num_pos, split_name, hard_ratio=HARD_NEG_RATIO,
                                guide_indices_for_hard=None, neg_ratio=NEG_POS_RATIO):
    global all_sampled_negs

    num_needed = int(num_pos * neg_ratio)
    negatives = []
    num_hard = int(num_needed * hard_ratio)

    print(f"\n  {split_name}: Sampling {num_needed:,} negatives ({neg_ratio}:1 ratio)...")

    # --- Hard negatives with seed-region awareness ---
    if guide_indices_for_hard is not None and num_hard > 0:
        hard_negs = []
        guides_list = list(set(guide_indices_for_hard))
        negs_per_guide = max(1, num_hard // len(guides_list) + 1)

        for g_idx in tqdm(guides_list, desc=f"{split_name} hard", leave=False):
            if len(hard_negs) >= num_hard:
                break
            guide_seq = guide_to_seq[g_idx]
            sample_size = min(1000, num_sites)
            candidate_sites = np.random.choice(num_sites, sample_size, replace=False)
            candidates = []
            for s_idx in candidate_sites:
                edge = (g_idx, int(s_idx))
                if edge in all_pos_set or edge in all_sampled_negs:
                    continue
                site_seq = site_to_seq[int(s_idx)]
                match_ratio = compute_match_ratio(guide_seq, site_seq)
                if MIN_MATCH_RATIO <= match_ratio <= MAX_MATCH_RATIO:
                    # Seed-region awareness: prioritize mismatches in seed (pos 1-12)
                    seed_matches = sum(1 for i in range(min(12, len(guide_seq), len(site_seq)))
                                       if guide_seq[i].upper() == site_seq[i].upper())
                    seed_ratio = seed_matches / min(12, len(guide_seq))
                    # Score: high overall match + low seed match = hardest
                    hardness = match_ratio * 0.4 + (1 - seed_ratio) * 0.6
                    candidates.append((int(s_idx), hardness))
            candidates.sort(key=lambda x: x[1], reverse=True)
            for s_idx, _ in candidates[:negs_per_guide]:
                if len(hard_negs) >= num_hard:
                    break
                edge = (g_idx, s_idx)
                hard_negs.append(edge)
                all_sampled_negs.add(edge)
        negatives.extend(hard_negs)
        print(f"    ✓ {len(hard_negs):,} hard negatives")

    # --- Random negatives ---
    num_random_needed = num_needed - len(negatives)
    random_negs = []
    attempts = 0
    max_attempts = num_random_needed * 200
    with tqdm(total=num_random_needed, desc=f"{split_name} random", leave=False) as pbar:
        while len(random_negs) < num_random_needed and attempts < max_attempts:
            g = np.random.randint(0, num_guides)
            s = np.random.randint(0, num_sites)
            edge = (g, s)
            if edge not in all_pos_set and edge not in all_sampled_negs:
                random_negs.append(edge)
                all_sampled_negs.add(edge)
                pbar.update(1)
            attempts += 1
    negatives.extend(random_negs)
    print(f"    ✓ {len(random_negs):,} random negatives")
    print(f"    ✓ Total: {len(negatives):,} negatives for {split_name}")
    return negatives


print("\n  Sampling negatives (disjoint across splits)...")
train_neg_list = sample_negatives_for_split(
    train_sup_pos_edges.shape[1], "TRAIN",
    guide_indices_for_hard=train_sup_pos_edges[0].tolist()
)
train_neg_edges = torch.tensor(train_neg_list, dtype=torch.long).T

val_neg_list = sample_negatives_for_split(
    val_pos_edges.shape[1], "VAL",
    guide_indices_for_hard=val_pos_edges[0].tolist()
)
val_neg_edges = torch.tensor(val_neg_list, dtype=torch.long).T

test_neg_list = sample_negatives_for_split(
    test_pos_edges.shape[1], "TEST",
    guide_indices_for_hard=test_pos_edges[0].tolist()
)
test_neg_edges = torch.tensor(test_neg_list, dtype=torch.long).T

# Verify negative disjointness
train_neg_set = set(train_neg_list)
val_neg_set = set(val_neg_list)
test_neg_set = set(test_neg_list)
assert len(train_neg_set & val_neg_set) == 0
assert len(train_neg_set & test_neg_set) == 0
assert len(val_neg_set & test_neg_set) == 0
print("  ✅ Negative edges are disjoint across splits")


# --- Step 4: Create HeteroData objects ---

def create_split_data(mp_edges_split, mp_features_split,
                      sup_pos_edges, sup_pos_features,
                      neg_edges, split_name):
    """Create HeteroData with separate MP and supervision edges.

    Phase 2 F2: MP edges != supervision edges for training.
    Phase 3 F11: Reverse edges carry mirrored edge features.
    """
    data = HeteroData()

    # Node features
    data['guide'].x = full_graph['guide'].x.clone()
    data['site'].x = full_graph['site'].x.clone()
    data['guide'].num_nodes = num_guides
    data['site'].num_nodes = num_sites

    # Message Passing edges
    data[edge_type].edge_index = mp_edges_split.clone()
    data[edge_type].edge_attr = mp_features_split.clone()

    # Phase 3 F11: Reverse edges WITH mirrored edge features
    data[rev_edge_type].edge_index = torch.stack(
        [mp_edges_split[1], mp_edges_split[0]], dim=0
    )
    data[rev_edge_type].edge_attr = mp_features_split.clone()

    # Compute features for negative edges
    print(f"  🔄 {split_name}: Computing features for {neg_edges.shape[1]:,} negative edges...")
    neg_features = compute_edge_features_for_edges(
        neg_edges, guide_to_seq, site_to_seq, guide_to_idx, site_to_idx, device='cpu'
    )

    # Supervision edges (positives + negatives)
    sup_edge_index = torch.cat([sup_pos_edges, neg_edges], dim=1)
    sup_labels = torch.cat([
        torch.ones(sup_pos_edges.shape[1], dtype=torch.float),
        torch.zeros(neg_edges.shape[1], dtype=torch.float)
    ])
    sup_features = torch.cat([sup_pos_features, neg_features], dim=0)

    # Shuffle supervision
    perm = torch.randperm(sup_labels.shape[0])
    data[edge_type].edge_label_index = sup_edge_index[:, perm]
    data[edge_type].edge_label = sup_labels[perm]
    data[edge_type].edge_label_attr = sup_features[perm]

    n_pos = int(sup_labels.sum().item())
    n_neg = int((sup_labels == 0).sum().item())
    print(f"  ✓ {split_name}: MP={mp_edges_split.shape[1]:,}, "
          f"Sup={n_pos + n_neg:,} ({n_pos:,} pos, {n_neg:,} neg, "
          f"ratio 1:{n_neg/max(n_pos,1):.1f})")
    return data


print("\n  Creating HeteroData objects...")

# TRAIN: MP = train_mp_edges (90%), Supervision = train_sup_edges (10%) + neg
train_data = create_split_data(
    mp_edges_split=mp_edges, mp_features_split=mp_features,
    sup_pos_edges=train_sup_pos_edges, sup_pos_features=train_sup_pos_features,
    neg_edges=train_neg_edges, split_name="TRAIN"
)

# VAL: MP = train_mp_edges, Supervision = val_pos + neg
val_data = create_split_data(
    mp_edges_split=mp_edges, mp_features_split=mp_features,
    sup_pos_edges=val_pos_edges, sup_pos_features=val_pos_features,
    neg_edges=val_neg_edges, split_name="VAL"
)

# TEST: MP = train_mp_edges, Supervision = test_pos + neg
test_data = create_split_data(
    mp_edges_split=mp_edges, mp_features_split=mp_features,
    sup_pos_edges=test_pos_edges, sup_pos_features=test_pos_features,
    neg_edges=test_neg_edges, split_name="TEST"
)


# --- Step 5: Verification ---
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

train_sup_edge_set = edge_index_to_set(train_data[edge_type].edge_label_index)
val_sup_edge_set = edge_index_to_set(val_data[edge_type].edge_label_index)
test_sup_edge_set = edge_index_to_set(test_data[edge_type].edge_label_index)

o_tv = len(train_sup_edge_set & val_sup_edge_set)
o_tt = len(train_sup_edge_set & test_sup_edge_set)
o_vt = len(val_sup_edge_set & test_sup_edge_set)

print(f"  Supervision overlap Train∩Val: {o_tv} {'✅' if o_tv == 0 else '❌'}")
print(f"  Supervision overlap Train∩Test: {o_tt} {'✅' if o_tt == 0 else '❌'}")
print(f"  Supervision overlap Val∩Test: {o_vt} {'✅' if o_vt == 0 else '❌'}")

# Check MP does not contain supervision positives
mp_edge_set = edge_index_to_set(mp_edges)
train_sup_pos_set = edge_index_to_set(train_sup_pos_edges)
val_in_mp = len(val_pos_set & mp_edge_set)
test_in_mp = len(test_pos_set & mp_edge_set)
train_sup_in_mp = len(train_sup_pos_set & mp_edge_set)

print(f"  Train supervision pos in MP: {train_sup_in_mp} {'✅' if train_sup_in_mp == 0 else '❌'}")
print(f"  Val pos in MP: {val_in_mp} {'✅' if val_in_mp == 0 else '❌'}")
print(f"  Test pos in MP: {test_in_mp} {'✅' if test_in_mp == 0 else '❌'}")

if INDUCTIVE_SPLIT:
    mp_guides = set(mp_edges[0].tolist())
    test_guides_in_mp = len(test_guide_set & mp_guides)
    val_guides_in_mp = len(val_guide_set & mp_guides)
    print(f"  Test guides in MP graph: {test_guides_in_mp} {'✅' if test_guides_in_mp == 0 else '❌'}")
    print(f"  Val guides in MP graph: {val_guides_in_mp} {'✅' if val_guides_in_mp == 0 else '❌'}")

assert o_tv == 0 and o_tt == 0 and o_vt == 0, "Supervision edge leakage!"
assert train_sup_in_mp == 0, "F2 violation: Train supervision edges found in MP!"
print("\n✅ All verification checks passed!")


# ============================================================================
# PHASE 3: ARCHITECTURE OVERHAUL
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 3: MODEL ARCHITECTURE")
print("=" * 70)


class EdgeFeatureEncoder(nn.Module):
    """Phase 3 F12: Learned edge feature encoder with BatchNorm."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class HeteroGNN_TransformerConv(nn.Module):
    """
    Phase 3: Enhanced Heterogeneous GNN using TransformerConv.

    Key improvements over HeteroGNN_SAGEConv:
    - F3:  TransformerConv with edge_dim for edge-aware message passing
    - F11: Reverse edges carry edge features via edge_attr_dict
    - F12: Edge feature encoder (BatchNorm + projection) before decoder
    - F13: Projection-based skip connections (never silently disabled)
    """

    def __init__(self, guide_in_channels, site_in_channels,
                 hidden_channels=64, out_channels=64,
                 edge_feat_dim=25, num_layers=3,
                 num_heads=4, dropout=0.20, edge_dropout=0.3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # --- Stage 1: Node Feature Encoders ---
        self.guide_encoder = nn.Sequential(
            Linear(guide_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_channels),
        )
        self.site_encoder = nn.Sequential(
            Linear(site_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_channels),
        )

        # --- Stage 1b: Edge Feature Encoder for MP ---
        # Project raw edge features to a dim compatible with TransformerConv
        self.mp_edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_channels),
            nn.ReLU(),
        )

        # --- Stage 2: TransformerConv layers with edge_dim ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # F13: explicit projection for skip

        for layer_i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if layer_i < num_layers - 1 else out_channels
            heads = num_heads
            if out_ch % heads != 0:
                raise ValueError(
                    f"Layer {layer_i}: out_channels={out_ch} must be divisible by "
                    f"num_heads={heads}"
                )
            head_dim = out_ch // heads

            conv_dict = {
                ('guide', 'targets', 'site'): TransformerConv(
                    (in_ch, in_ch), head_dim,
                    heads=heads, edge_dim=hidden_channels,
                    dropout=dropout, concat=True,
                ),
                ('site', 'rev_targets', 'guide'): TransformerConv(
                    (in_ch, in_ch), head_dim,
                    heads=heads, edge_dim=hidden_channels,
                    dropout=dropout, concat=True,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            self.norms.append(nn.ModuleDict({
                'guide': nn.LayerNorm(out_ch),
                'site': nn.LayerNorm(out_ch),
            }))

            # F13: Projection for skip connection if dims differ
            if in_ch != out_ch:
                self.skip_projs.append(nn.ModuleDict({
                    'guide': nn.Linear(in_ch, out_ch),
                    'site': nn.Linear(in_ch, out_ch),
                }))
            else:
                self.skip_projs.append(None)

        # --- Stage 3: Edge Feature Encoder for Decoder (F12) ---
        self.edge_feat_encoder = EdgeFeatureEncoder(edge_feat_dim, hidden_channels)

        # --- Stage 4: Edge Decoder ---
        decoder_input_dim = out_channels + out_channels + hidden_channels
        self.edge_decoder = nn.Sequential(
            Linear(decoder_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(edge_dropout),
            nn.LayerNorm(128),
            Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(edge_dropout),
            nn.LayerNorm(64),
            Linear(64, 1),
        )

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Encode with edge-aware message passing."""
        h_dict = {
            'guide': self.guide_encoder(x_dict['guide']),
            'site': self.site_encoder(x_dict['site']),
        }

        # Encode MP edge features if available
        encoded_edge_attr_dict = {}
        if edge_attr_dict is not None:
            for et, ea in edge_attr_dict.items():
                encoded_edge_attr_dict[et] = self.mp_edge_encoder(ea)

        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            h_dict_prev = {k: v for k, v in h_dict.items()}

            # HeteroConv requires kwargs ending in '_dict'. It strips the
            # suffix and passes edge_attr=<per-edge-type tensor> to each
            # sub-conv (TransformerConv), matching its forward() signature.
            if encoded_edge_attr_dict:
                h_dict = conv(h_dict, edge_index_dict,
                              edge_attr_dict=encoded_edge_attr_dict)
            else:
                h_dict = conv(h_dict, edge_index_dict)

            for node_type in h_dict.keys():
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = F.dropout(
                    h_dict[node_type], p=self.dropout, training=self.training
                )
                # F13: Skip connection with projection if needed
                skip_proj = self.skip_projs[i]
                if skip_proj is not None:
                    h_prev = skip_proj[node_type](h_dict_prev[node_type])
                else:
                    h_prev = h_dict_prev[node_type]
                h_dict[node_type] = h_dict[node_type] + h_prev
                h_dict[node_type] = norm_dict[node_type](h_dict[node_type])

        return h_dict

    def decode(self, h_dict, edge_label_index, edge_features):
        guide_idx = edge_label_index[0]
        site_idx = edge_label_index[1]
        guide_emb = h_dict['guide'][guide_idx]
        site_emb = h_dict['site'][site_idx]

        # F12: encode edge features before decoder
        encoded_ef = self.edge_feat_encoder(edge_features)
        edge_input = torch.cat([guide_emb, site_emb, encoded_ef], dim=-1)
        return self.edge_decoder(edge_input)

    def forward(self, x_dict, edge_index_dict, edge_label_index,
                edge_features, edge_attr_dict=None):
        h_dict = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        out = self.decode(h_dict, edge_label_index, edge_features)
        return out.squeeze(-1)


print("✓ Model class defined (HeteroGNN_TransformerConv)")


# ============================================================================
# PHASE 4: LOSS FUNCTION (FocalLoss replaces label-smoothed BCE)
# ============================================================================

class FocalLoss(nn.Module):
    """Phase 4 F10: Focal Loss replaces label-smoothed BCE + pos_weight.

    Handles both class imbalance (via alpha) and hard example focusing
    (via gamma) in a single, mathematically consistent loss function.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def prepare_edge_index_dict(data):
    edge_index_dict = {}
    for et in data.edge_types:
        if hasattr(data[et], 'edge_index'):
            ei = data[et].edge_index
            if ei is not None and ei.numel() > 0:
                edge_index_dict[et] = ei
    return edge_index_dict


def prepare_edge_attr_dict(data):
    """Prepare edge_attr_dict for TransformerConv (Phase 3 F3)."""
    edge_attr_dict = {}
    for et in data.edge_types:
        if hasattr(data[et], 'edge_attr'):
            ea = data[et].edge_attr
            if ea is not None and ea.numel() > 0:
                edge_attr_dict[et] = ea
    return edge_attr_dict


def train_epoch(model, data, optimizer, criterion, device, edge_features=None):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)

    edge_label_index = data['guide', 'targets', 'site'].edge_label_index
    edge_labels = data['guide', 'targets', 'site'].edge_label.float()

    if edge_features is not None:
        edge_features = edge_features.to(device)

    edge_index_dict = prepare_edge_index_dict(data)
    edge_attr_dict = prepare_edge_attr_dict(data)

    # Move edge_attr_dict to device
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

    out = model(data.x_dict, edge_index_dict, edge_label_index,
                edge_features=edge_features, edge_attr_dict=edge_attr_dict)

    loss = criterion(out, edge_labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, device, edge_features=None,
             threshold=CALIBRATED_THRESHOLD, return_preds=False):
    model.eval()
    data = data.to(device)

    edge_label_index = data['guide', 'targets', 'site'].edge_label_index
    edge_labels = data['guide', 'targets', 'site'].edge_label.float()

    if edge_features is not None:
        edge_features = edge_features.to(device)

    edge_index_dict = prepare_edge_index_dict(data)
    edge_attr_dict = prepare_edge_attr_dict(data)
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

    out = model(data.x_dict, edge_index_dict, edge_label_index,
                edge_features=edge_features, edge_attr_dict=edge_attr_dict)

    loss = criterion(out, edge_labels)

    y_true = edge_labels.cpu().numpy()
    y_pred_probs = torch.sigmoid(out).cpu().numpy()
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    metrics = {'loss': loss.item(), 'threshold': threshold}
    try:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        metrics['auroc'] = 0.0
    try:
        metrics['auprc'] = average_precision_score(y_true, y_pred_probs)
    except Exception:
        metrics['auprc'] = 0.0
    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary)
    except Exception:
        metrics['f1'] = 0.0
    try:
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    except Exception:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
    metrics['n_pos'] = int(y_true.sum())
    metrics['n_neg'] = int((1 - y_true).sum())

    if return_preds:
        return metrics, y_true, y_pred_probs, y_pred_binary
    return metrics


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix} Metrics:")
    print("=" * 50)
    print(f"  Loss:            {metrics['loss']:.4f}")
    print(f"  AUPRC (primary): {metrics['auprc']:.4f}")
    print(f"  AUROC:           {metrics['auroc']:.4f}")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  Threshold:       {metrics['threshold']:.4f}")
    print(f"  Pos/Neg:         {metrics['n_pos']:,}/{metrics['n_neg']:,}")


# ============================================================================
# TRAINING LOOP (with all Phase 4 fixes)
# ============================================================================

def run_training(model, train_data, val_data, criterion, device, tag='full'):
    """Run training loop with early stopping, proper checkpointing (F9)."""

    GUIDE_IN_CHANNELS = metadata_info['guide_feature_dim']
    SITE_IN_CHANNELS = metadata_info['site_feature_dim']

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'val_auprc': [], 'val_auroc': [], 'val_f1': [],
        'best_epoch': 0, 'overfitting_gap': [],
    }

    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    # Hyperparameters dict for checkpoint (F9)
    hparams = {
        'hidden_channels': HIDDEN_CHANNELS, 'out_channels': OUT_CHANNELS,
        'edge_feat_dim': EDGE_FEAT_DIM, 'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS, 'dropout': DROPOUT, 'edge_dropout': EDGE_DROPOUT,
        'guide_in_channels': GUIDE_IN_CHANNELS, 'site_in_channels': SITE_IN_CHANNELS,
        'neg_pos_ratio': NEG_POS_RATIO,
        'inductive_split': INDUCTIVE_SPLIT,
    }

    train_edge_feat = train_data['guide', 'targets', 'site'].edge_label_attr
    val_edge_feat = val_data['guide', 'targets', 'site'].edge_label_attr

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {type(model).__name__} | Params: {total_params:,}")
    print(f"  Loss: FocalLoss(alpha=0.25, gamma=2.0)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts(T_0=20)")
    print(f"  Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}, Grad clip: {GRAD_CLIP}")
    print("-" * 90)

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_data, optimizer, criterion, device,
                     edge_features=train_edge_feat)
        val_metrics = evaluate(model, val_data, criterion, device,
                       edge_features=val_edge_feat)

        overfitting_gap = abs(train_loss - val_metrics['loss'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_f1'].append(val_metrics['f1'])
        history['overfitting_gap'].append(overfitting_gap)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        improvement = ""
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history['best_epoch'] = epoch
            improvement = "✓ BEST"

            # F9: Save best checkpoint with hyperparameters
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'hparams': hparams,
                'val_auprc': val_metrics['auprc'],
                'val_auroc': val_metrics['auroc'],
                'val_f1': val_metrics['f1'],
                'history': history,
                'best_val_f1': best_val_f1,
            }, CHECKPOINT_DIR / f'best_{tag}.pt')
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        gap_flag = "⚠️" if overfitting_gap > 0.3 else ""

        if epoch % 1 == 0 or improvement or epoch <= 3:
            print(f"  Ep {epoch:03d}/{NUM_EPOCHS} | "
                  f"TrL:{train_loss:.4f} VaL:{val_metrics['loss']:.4f} "
                  f"Gap:{overfitting_gap:.3f}{gap_flag} | "
                  f"AUPRC:{val_metrics['auprc']:.4f} "
                  f"AUROC:{val_metrics['auroc']:.4f} "
                  f"F1:{val_metrics['f1']:.4f} | "
                  f"LR:{current_lr:.1e} {epoch_time:.1f}s {improvement}")

        # if patience_counter >= PATIENCE:
        #     print(f"\n  ⚠️ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        #     break

    total_time = time.time() - start_time
    print(f"\n  Training complete: {total_time / 60:.1f} min, "
            f"Best epoch: {history['best_epoch']}, Best F1: {best_val_f1:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, best_val_f1


# ============================================================================
# PHASE 5: EVALUATION & INTERPRETABILITY
# ============================================================================

@torch.no_grad()
def per_guide_evaluation(model, data, device, edge_features=None):
    """Phase 5: Per-guide AUPRC breakdown."""
    model.eval()
    data = data.to(device)

    edge_label_index = data['guide', 'targets', 'site'].edge_label_index
    edge_labels = data['guide', 'targets', 'site'].edge_label.float()

    if edge_features is not None:
        edge_features = edge_features.to(device)

    edge_index_dict = prepare_edge_index_dict(data)
    edge_attr_dict = prepare_edge_attr_dict(data)
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

    out = model(data.x_dict, edge_index_dict, edge_label_index,
                edge_features=edge_features, edge_attr_dict=edge_attr_dict)

    y_true = edge_labels.cpu().numpy()
    y_pred = torch.sigmoid(out).cpu().numpy()
    guide_indices = edge_label_index[0].cpu().numpy()

    per_guide = {}
    for g_idx in np.unique(guide_indices):
        mask = guide_indices == g_idx
        g_true = y_true[mask]
        g_pred = y_pred[mask]
        if len(np.unique(g_true)) < 2:
            continue
        try:
            auprc = average_precision_score(g_true, g_pred)
            auroc = roc_auc_score(g_true, g_pred)
        except Exception:
            continue
        guide_name = guide_to_seq.get(g_idx, f"guide_{g_idx}")
        per_guide[guide_name] = {
            'auprc': auprc, 'auroc': auroc,
            'n_pos': int(g_true.sum()), 'n_neg': int((1 - g_true).sum()),
        }

    return per_guide


def calibrate_threshold(model, val_data, criterion, device, edge_features=None):
    """Optional utility: estimate a threshold from the validation PR curve."""
    metrics, y_true, y_pred_probs, _ = evaluate(
        model, val_data, criterion, device, edge_features=edge_features,
        return_preds=True
    )
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred_probs)
    # Optimal = max F1
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / \
                (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  Calibrated threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
    return best_threshold


# ============================================================================
# MAIN EXECUTION
# ============================================================================

GUIDE_IN_CHANNELS = metadata_info['guide_feature_dim']
SITE_IN_CHANNELS = metadata_info['site_feature_dim']

# Phase 4 F10: FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# ---- Full model training ----
print("\n" + "=" * 70)
print("FULL MODEL TRAINING")
print("=" * 70)

print("\n--- Full Model: TransformerConv + Edge Features ---")
model_full = HeteroGNN_TransformerConv(
    guide_in_channels=GUIDE_IN_CHANNELS, site_in_channels=SITE_IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS,
    edge_feat_dim=EDGE_FEAT_DIM, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, dropout=DROPOUT, edge_dropout=EDGE_DROPOUT,
).to(device)

history_full, best_f1_full = run_training(
    model_full, train_data, val_data, criterion, device,
    tag='full'
)


# ============================================================================
# FINAL EVALUATION ON TEST SET
# ============================================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION (Best Full Model)")
print("=" * 70)

# Load best saved checkpoint from disk for test inference
best_ckpt_path = CHECKPOINT_DIR / 'best_full.pt'
best_ckpt = torch.load(best_ckpt_path, map_location=device)
model_full.load_state_dict(best_ckpt['model_state_dict'])
model_full.eval()
print(f"  ✓ Loaded best checkpoint from: {best_ckpt_path}")
print(f"  ✓ Checkpoint epoch: {best_ckpt.get('epoch', 'N/A')}, "
    f"Val F1: {best_ckpt.get('best_val_f1', best_ckpt.get('val_f1', float('nan'))):.4f}")

test_edge_feat = test_data['guide', 'targets', 'site'].edge_label_attr
test_metrics, y_true, y_pred_probs, y_pred_binary = evaluate(
    model_full, test_data, criterion, device,
    edge_features=test_edge_feat, return_preds=True
)
print_metrics(test_metrics, "Test (Full Model)")


# ============================================================================
# PHASE 5: PER-GUIDE EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 5: PER-GUIDE AUPRC BREAKDOWN (Test Set)")
print("=" * 70)

per_guide = per_guide_evaluation(
    model_full, test_data, device,
    edge_features=test_edge_feat
)

if per_guide:
    auprc_values = [v['auprc'] for v in per_guide.values()]
    print(f"  Guides evaluated: {len(per_guide)}")
    print(f"  AUPRC mean: {np.mean(auprc_values):.4f} ± {np.std(auprc_values):.4f}")
    print(f"  AUPRC median: {np.median(auprc_values):.4f}")
    print(f"  AUPRC min: {np.min(auprc_values):.4f}, max: {np.max(auprc_values):.4f}")

    # Show worst 5 guides
    sorted_guides = sorted(per_guide.items(), key=lambda x: x[1]['auprc'])
    print(f"\n  Bottom 5 guides (lowest AUPRC):")
    for guide_name, stats in sorted_guides[:5]:
        print(f"    {guide_name[:30]:<32} AUPRC={stats['auprc']:.4f} "
              f"(pos={stats['n_pos']}, neg={stats['n_neg']})")
else:
    print("  ⚠️ No guides with both pos/neg edges in test set")

print("\n--- Fixed Calibrated Operating Threshold ---")
best_threshold = CALIBRATED_THRESHOLD
y_pred_calibrated = (y_pred_probs >= best_threshold).astype(int)
cal_f1 = f1_score(y_true, y_pred_calibrated)
cal_prec = precision_score(y_true, y_pred_calibrated, zero_division=0)
cal_rec = recall_score(y_true, y_pred_calibrated, zero_division=0)
print(f"  Test with fixed calibrated threshold ({best_threshold:.4f}):")
print(f"    F1={cal_f1:.4f}, Precision={cal_prec:.4f}, Recall={cal_rec:.4f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Training & Validation Loss (full model)
h = history_full
axes[0, 0].plot(h['train_loss'], label='Training Loss', color='#2563eb', linewidth=2)
axes[0, 0].plot(h['val_loss'], label='Validation Loss', color='#dc2626', linewidth=2)
axes[0, 0].axvline(h['best_epoch'], color='green', linestyle='--',
                    label=f'Best Epoch ({h["best_epoch"]})', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Validation AUPRC (full model)
axes[0, 1].plot(h['val_auprc'], label='Validation AUPRC', color='#16a34a', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUPRC')
axes[0, 1].set_title('Validation AUPRC (Full Model)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Validation AUROC (full model)
axes[0, 2].plot(h['val_auroc'], label='Validation AUROC', color='#7c3aed', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('AUROC')
axes[0, 2].set_title('Validation AUROC (Full Model)', fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Overfitting gap
axes[1, 0].plot(h['overfitting_gap'], color='#7c3aed', linewidth=2)
axes[1, 0].axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
axes[1, 0].axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Severe (>0.3)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('|Train Loss - Val Loss|')
axes[1, 0].set_title('Overfitting Gap', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Precision-Recall Curve (test)
precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_probs)
axes[1, 1].plot(recall_arr, precision_arr, color='#16a34a', linewidth=2)
axes[1, 1].fill_between(recall_arr, precision_arr, alpha=0.2, color='#16a34a')
axes[1, 1].scatter(
    [cal_rec], [cal_prec], color='orange', s=60,
    label=f'Fixed calibrated operating point ({best_threshold:.4f})', zorder=5
)
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title(f'PR Curve (Test) AUPRC={test_metrics["auprc"]:.4f}', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. ROC Curve (test)
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
axes[1, 2].plot(fpr, tpr, color='#dc2626', linewidth=2)
axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
axes[1, 2].fill_between(fpr, tpr, alpha=0.2, color='#dc2626')
axes[1, 2].set_xlabel('False Positive Rate')
axes[1, 2].set_ylabel('True Positive Rate')
axes[1, 2].set_title(f'ROC Curve (Test) AUROC={test_metrics["auroc"]:.4f}', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'offtarget_enhanced_training.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Training plots saved")

# Prediction distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_pred_probs[y_true == 1], bins=50, alpha=0.6, label='True OFF-targets',
             color='#dc2626', edgecolor='black')
axes[0].hist(y_pred_probs[y_true == 0], bins=50, alpha=0.6, label='True Non-targets',
             color='#2563eb', edgecolor='black')
axes[0].axvline(best_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Fixed calibrated ({best_threshold:.4f})')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Prediction Distribution (Test Set)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot([y_pred_probs[y_true == 0], y_pred_probs[y_true == 1]],
                labels=['Non-targets', 'OFF-targets'],
                patch_artist=True,
                boxprops=dict(facecolor='#3b82f6', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].axhline(best_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Fixed calibrated ({best_threshold:.4f})', alpha=0.7)
axes[1].set_ylabel('Predicted Probability')
axes[1].set_title('Prediction by Class', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIG_DIR / 'offtarget_enhanced_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Distribution plots saved")

# Per-guide AUPRC histogram
if per_guide:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(auprc_values, bins=30, color='#16a34a', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(auprc_values), color='red', linestyle='--',
               label=f'Mean: {np.mean(auprc_values):.4f}')
    ax.axvline(np.median(auprc_values), color='blue', linestyle='--',
               label=f'Median: {np.median(auprc_values):.4f}')
    ax.set_xlabel('Per-Guide AUPRC')
    ax.set_ylabel('Count')
    ax.set_title('Per-Guide AUPRC Distribution (Test Set)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'offtarget_per_guide_auprc.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Per-guide AUPRC plot saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Split type: {'Inductive (guide-level)' if INDUCTIVE_SPLIT else 'Transductive (edge-level)'}")
print(f"  Neg:Pos ratio: {NEG_POS_RATIO}:1")
print(f"  MP/Supervision separation: {MP_SUPERVISION_RATIO:.0%} MP / {1-MP_SUPERVISION_RATIO:.0%} supervision")
print(f"  Architecture: TransformerConv with {NUM_HEADS} heads, {NUM_LAYERS} layers")
print(f"  Edge features in MP: Yes (edge_dim={EDGE_FEAT_DIM})")
print(f"  Loss: FocalLoss(alpha=0.25, gamma=2.0)")
print()
print(f"  Test Results (Full Model):")
print(f"    AUPRC: {test_metrics['auprc']:.4f}")
print(f"    AUROC: {test_metrics['auroc']:.4f}")
print(f"    F1:    {test_metrics['f1']:.4f}")
print(f"    Fixed calibrated F1: {cal_f1:.4f} (threshold={best_threshold:.4f})")
print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
print(f"  Figures saved to: {FIG_DIR}")
print("=" * 70)
