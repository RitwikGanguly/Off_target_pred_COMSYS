#!/usr/bin/env python3
"""Build the heterogeneous bipartite graph used by offtarget_pred_model.py.

This script prepares:
1) `data/offtarget_filtered.csv` (OFF + Homo sapiens subset)
2) `model/hetero_graph_data_new1.pt` (HeteroData graph + metadata)

Data source files are expected in `Off_target_pred_COMSYS/data/`.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from Bio import pairwise2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "model"

RAW_DATA_PATH = Path(
    os.getenv("OFFTARGET_RAW_DATA_PATH", str(DATA_DIR / "allframe_update_addEpige.txt"))
)
FILTERED_CSV_PATH = Path(
    os.getenv("OFFTARGET_FILTERED_CSV_PATH", str(DATA_DIR / "offtarget_filtered.csv"))
)
GRAPH_OUTPUT_PATH = Path(
    os.getenv("OFFTARGET_GRAPH_DATA_PATH", str(MODEL_DIR / "hetero_graph_data_new1.pt"))
)

SEED = int(os.getenv("OFFTARGET_SEED", "42"))
TARGET_EMBED_DIM = int(os.getenv("OFFTARGET_TARGET_EMBED_DIM", "128"))

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Sequence and feature utilities
# -----------------------------------------------------------------------------

def sanitize_seq(seq: str) -> str:
    seq_u = str(seq).upper()
    allowed = set("ACGTUN")
    return "".join(ch if ch in allowed else "N" for ch in seq_u)


def compute_gc_content(seq: str) -> float:
    seq_u = sanitize_seq(seq)
    if len(seq_u) == 0:
        return 0.0
    gc = seq_u.count("G") + seq_u.count("C")
    return gc / len(seq_u)


def compute_secondary_structure_stability(seq: str) -> float:
    """Return normalized stability score in [0, 1]."""
    seq_u = sanitize_seq(seq)
    rna_seq = seq_u.replace("T", "U")

    # Try ViennaRNA first
    try:
        import RNA

        _, mfe = RNA.fold(rna_seq)
        return max(0.0, min(1.0, (-mfe) / 50.0))
    except Exception:
        pass

    # Fallback: simple GC + di-nucleotide heuristic
    gc = compute_gc_content(seq_u)
    stable_pairs = 0
    for i in range(max(len(seq_u) - 1, 0)):
        if seq_u[i : i + 2] in {"GC", "CG", "GG", "CC"}:
            stable_pairs += 1
    if len(seq_u) > 1:
        pair_term = stable_pairs / (len(seq_u) - 1)
    else:
        pair_term = gc
    return (gc * 0.7) + (pair_term * 0.3)


def align_sequences(guide_seq: str, target_seq: str) -> float:
    alignments = pairwise2.align.globalxx(guide_seq, target_seq)
    if not alignments:
        return 0.0
    max_score = alignments[0].score
    denom = min(len(guide_seq), len(target_seq))
    return max_score / denom if denom > 0 else 0.0


def compute_mismatch_vector(guide_seq: str, target_seq: str) -> tuple[list[int], int]:
    guide = sanitize_seq(guide_seq)
    target = sanitize_seq(target_seq)
    max_len = max(len(guide), len(target))
    guide_padded = guide.ljust(max_len, "N")
    target_padded = target.ljust(max_len, "N")

    match_vector: list[int] = []
    mismatch_count = 0
    for i in range(max_len):
        if guide_padded[i] == target_padded[i] and guide_padded[i] != "N":
            match_vector.append(1)
        else:
            match_vector.append(0)
            mismatch_count += 1
    return match_vector, mismatch_count


def compute_weighted_mismatch_score(guide_seq: str, target_seq: str) -> float:
    match_vector, _ = compute_mismatch_vector(guide_seq, target_seq)
    length = len(match_vector)
    weights = np.linspace(0.5, 2.0, length) if length > 0 else np.array([1.0])
    weighted = sum((1 - m) * w for m, w in zip(match_vector, weights))
    denom = float(np.sum(weights))
    return weighted / denom if denom > 0 else 0.0


def compute_cfd_like_score(guide_seq: str, target_seq: str) -> float:
    match_vector, _ = compute_mismatch_vector(guide_seq, target_seq)
    length = len(match_vector)
    weights = np.linspace(0.5, 1.5, length) if length > 0 else np.array([1.0])
    score = sum(m * w for m, w in zip(match_vector, weights))
    denom = float(np.sum(weights))
    return score / denom if denom > 0 else 0.0


def compute_melting_temperature(guide_seq: str, target_seq: str) -> float:
    seq = sanitize_seq(guide_seq) if len(guide_seq) <= len(target_seq) else sanitize_seq(target_seq)
    return 2 * (seq.count("A") + seq.count("T")) + 4 * (seq.count("G") + seq.count("C"))


def expand_bio_features(gc: float, stability: float) -> np.ndarray:
    return np.array(
        [
            gc,
            stability,
            gc**2,
            stability**2,
            gc * stability,
            abs(gc - 0.5),
            np.log(gc + 1e-6),
            np.log(stability + 1e-6),
        ],
        dtype=np.float32,
    )


def get_kmer_embedding(seq: str, k: int, embed_dim: int = 128) -> np.ndarray:
    seq_u = sanitize_seq(seq)
    kmers: dict[str, int] = {}
    for i in range(max(len(seq_u) - k + 1, 0)):
        token = seq_u[i : i + k]
        kmers[token] = kmers.get(token, 0) + 1

    emb = np.zeros(embed_dim, dtype=np.float32)
    for token, count in kmers.items():
        stable_idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % embed_dim
        emb[stable_idx] += float(count)

    total = float(emb.sum())
    if total > 0:
        emb /= total
    return emb


def get_rna_embedding(seq: str, model, tokenizer, device: torch.device) -> np.ndarray:
    if model is None or tokenizer is None:
        return get_kmer_embedding(seq, k=3, embed_dim=128)

    try:
        with torch.no_grad():
            rna_seq = sanitize_seq(seq).replace("T", "U")
            inputs = tokenizer(
                rna_seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception:
        return get_kmer_embedding(seq, k=3, embed_dim=128)


def get_dna_embedding(seq: str, model, tokenizer, device: torch.device) -> np.ndarray:
    if model is None or tokenizer is None:
        return get_kmer_embedding(seq, k=4, embed_dim=128)

    try:
        with torch.no_grad():
            inputs = tokenizer(
                sanitize_seq(seq),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception:
        return get_kmer_embedding(seq, k=4, embed_dim=128)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def load_and_filter_raw_data(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Input data not found: {raw_path}\n"
            "Expected file in Off_target_pred_COMSYS/data/allframe_update_addEpige.txt"
        )

    df = pd.read_csv(raw_path, sep="\t")
    filtered = df.loc[
        (df["Identity"] == "OFF") & (df["Species"] == "Homo sapiens")
    ].copy()
    filtered = filtered[
        [
            "Guide_sequence",
            "Target_sequence",
            "PAM",
            "Target_region",
            "Identity",
            "assembly_target_region",
            "gRNA",
        ]
    ]
    filtered = filtered.drop_duplicates().reset_index(drop=True)

    filtered["Guide_sequence"] = filtered["Guide_sequence"].astype(str).map(sanitize_seq)
    filtered["Target_sequence"] = filtered["Target_sequence"].astype(str).map(sanitize_seq)

    return filtered


def load_foundation_models(device: torch.device):
    rna_model = None
    rna_tokenizer = None
    dna_model = None
    dna_tokenizer = None

    print("\nLoading sequence foundation models (with k-mer fallback)...")

    try:
        from multimolecule import RnaTokenizer

        rna_tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
        rna_model = AutoModel.from_pretrained("multimolecule/rnafm", trust_remote_code=True)
        rna_model.eval().to(device)
        print("  RNA-FM: loaded")
    except Exception as exc:
        print(f"  RNA-FM unavailable ({exc}); using k-mer fallback")

    try:
        config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        dna_tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        dna_model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", config=config, trust_remote_code=True
        )
        dna_model.eval().to(device)
        print("  DNABERT-2: loaded")
    except Exception as exc:
        print(f"  DNABERT-2 unavailable ({exc}); using k-mer fallback")

    return rna_model, rna_tokenizer, dna_model, dna_tokenizer


def build_node_features(
    df_filtered: pd.DataFrame,
    rna_model,
    rna_tokenizer,
    dna_model,
    dna_tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int], dict[int, str]]:
    unique_guides = df_filtered["Guide_sequence"].unique()
    unique_sites = df_filtered["Target_region"].unique()

    guide_to_idx = {guide: idx for idx, guide in enumerate(unique_guides)}
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}

    # Site-region to representative sequence
    region_to_seq = (
        df_filtered[["Target_region", "Target_sequence"]]
        .drop_duplicates("Target_region")
        .set_index("Target_region")["Target_sequence"]
        .to_dict()
    )

    print(f"\nGuides: {len(unique_guides)}, Sites: {len(unique_sites)}, OFF edges: {len(df_filtered)}")

    guide_embeddings_list = []
    guide_bio_list = []
    for guide_seq in tqdm(unique_guides, desc="Guide embeddings"):
        emb = get_rna_embedding(guide_seq, rna_model, rna_tokenizer, device)
        guide_embeddings_list.append(emb)
        gc = compute_gc_content(guide_seq)
        stab = compute_secondary_structure_stability(guide_seq)
        guide_bio_list.append((gc, stab))

    site_embeddings_list = []
    site_bio_list = []
    site_idx_to_seq: dict[int, str] = {}
    for site_region in tqdm(unique_sites, desc="Site embeddings"):
        site_idx = site_to_idx[site_region]
        target_seq = region_to_seq.get(site_region, "ATCG")
        site_idx_to_seq[site_idx] = target_seq

        emb = get_dna_embedding(target_seq, dna_model, dna_tokenizer, device)
        site_embeddings_list.append(emb)
        gc = compute_gc_content(target_seq)
        stab = compute_secondary_structure_stability(target_seq)
        site_bio_list.append((gc, stab))

    guide_emb = np.asarray(guide_embeddings_list)
    site_emb = np.asarray(site_embeddings_list)

    guide_target_dim = min(TARGET_EMBED_DIM, guide_emb.shape[1])
    site_target_dim = min(TARGET_EMBED_DIM, site_emb.shape[1])

    if guide_emb.shape[1] > guide_target_dim:
        guide_emb = PCA(n_components=guide_target_dim, random_state=SEED).fit_transform(guide_emb)
    if site_emb.shape[1] > site_target_dim:
        site_emb = PCA(n_components=site_target_dim, random_state=SEED).fit_transform(site_emb)

    guide_features = []
    for i, (gc, stab) in enumerate(guide_bio_list):
        guide_features.append(np.concatenate([guide_emb[i], expand_bio_features(gc, stab)]))

    site_features = []
    for i, (gc, stab) in enumerate(site_bio_list):
        site_features.append(np.concatenate([site_emb[i], expand_bio_features(gc, stab)]))

    guide_features = StandardScaler().fit_transform(np.asarray(guide_features)).astype(np.float32)
    site_features = StandardScaler().fit_transform(np.asarray(site_features)).astype(np.float32)

    return guide_features, site_features, guide_to_idx, site_to_idx, site_idx_to_seq


def build_edge_features(
    df_filtered: pd.DataFrame,
    guide_to_idx: dict[str, int],
    site_to_idx: dict[str, int],
    site_idx_to_seq: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_list = []
    edge_labels = []

    for _, row in df_filtered.iterrows():
        guide_seq = row["Guide_sequence"]
        site_region = row["Target_region"]
        if pd.isna(guide_seq) or pd.isna(site_region):
            continue
        edge_list.append([guide_to_idx[guide_seq], site_to_idx[site_region]])
        edge_labels.append(1)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(edge_labels, dtype=torch.long)

    guide_idx_to_seq = {idx: seq for seq, idx in guide_to_idx.items()}
    edge_features_list = []

    for i in tqdm(range(edge_index.shape[1]), desc="Edge features"):
        g_idx = int(edge_index[0, i])
        s_idx = int(edge_index[1, i])
        guide_seq = guide_idx_to_seq[g_idx]
        target_seq = site_idx_to_seq[s_idx]

        match_vector, total_mismatch = compute_mismatch_vector(guide_seq, target_seq)
        if len(match_vector) < 20:
            match_vector = match_vector + [0] * (20 - len(match_vector))
        else:
            match_vector = match_vector[:20]

        mm_norm = total_mismatch / max(len(guide_seq), len(target_seq)) if max(len(guide_seq), len(target_seq)) > 0 else 0.0
        align_score = align_sequences(guide_seq, target_seq)
        weighted_mm = compute_weighted_mismatch_score(guide_seq, target_seq)
        cfd_score = compute_cfd_like_score(guide_seq, target_seq)
        tm_norm = compute_melting_temperature(guide_seq, target_seq) / 100.0

        edge_features_list.append(match_vector + [mm_norm, align_score, weighted_mm, cfd_score, tm_norm])

    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    return edge_index, edge_attr, edge_label


def build_and_save_graph(
    guide_features: np.ndarray,
    site_features: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_label: torch.Tensor,
    guide_to_idx: dict[str, int],
    site_to_idx: dict[str, int],
    output_path: Path,
) -> None:
    data = HeteroData()
    data["guide"].x = torch.tensor(guide_features, dtype=torch.float)
    data["guide"].num_nodes = guide_features.shape[0]
    data["site"].x = torch.tensor(site_features, dtype=torch.float)
    data["site"].num_nodes = site_features.shape[0]

    data[("guide", "targets", "site")].edge_index = edge_index
    data[("guide", "targets", "site")].edge_attr = edge_attr
    data[("guide", "targets", "site")].edge_label = edge_label

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "full_graph": data,
            "guide_to_idx": guide_to_idx,
            "site_to_idx": site_to_idx,
            "metadata": {
                "num_guides": guide_features.shape[0],
                "num_sites": site_features.shape[0],
                "num_observed_edges": int(edge_index.shape[1]),
                "guide_feature_dim": int(guide_features.shape[1]),
                "site_feature_dim": int(site_features.shape[1]),
                "edge_feature_dim": int(edge_attr.shape[1]),
            },
        },
        output_path,
    )

    print("\nGraph summary")
    print("-" * 70)
    print(data)
    print(f"Guides: {guide_features.shape[0]}")
    print(f"Sites: {site_features.shape[0]}")
    print(f"Observed OFF edges: {edge_index.shape[1]}")
    print(f"Guide feature dim: {guide_features.shape[1]}")
    print(f"Site feature dim: {site_features.shape[1]}")
    print(f"Edge feature dim: {edge_attr.shape[1]}")
    print(f"Saved graph -> {output_path}")


def main() -> None:
    print("=" * 70)
    print("CRISPR OFF-TARGET GRAPH BUILDING")
    print("=" * 70)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Raw input: {RAW_DATA_PATH}")
    print(f"Filtered CSV output: {FILTERED_CSV_PATH}")
    print(f"Graph output: {GRAPH_OUTPUT_PATH}")
    print(f"Device: {DEVICE}")

    df_filtered = load_and_filter_raw_data(RAW_DATA_PATH)
    FILTERED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(FILTERED_CSV_PATH, index=False)
    print(f"\nSaved filtered OFF-target CSV -> {FILTERED_CSV_PATH} ({len(df_filtered)} rows)")

    rna_model, rna_tokenizer, dna_model, dna_tokenizer = load_foundation_models(DEVICE)

    (
        guide_features,
        site_features,
        guide_to_idx,
        site_to_idx,
        site_idx_to_seq,
    ) = build_node_features(
        df_filtered,
        rna_model,
        rna_tokenizer,
        dna_model,
        dna_tokenizer,
        DEVICE,
    )

    edge_index, edge_attr, edge_label = build_edge_features(
        df_filtered,
        guide_to_idx,
        site_to_idx,
        site_idx_to_seq,
    )

    build_and_save_graph(
        guide_features,
        site_features,
        edge_index,
        edge_attr,
        edge_label,
        guide_to_idx,
        site_to_idx,
        GRAPH_OUTPUT_PATH,
    )

    print("\nDone. Next step: run `python model/offtarget_pred_model.py`")


if __name__ == "__main__":
    main()
