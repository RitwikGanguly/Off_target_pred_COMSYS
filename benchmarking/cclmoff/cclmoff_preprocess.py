#!/usr/bin/env python3
"""
CCLMoff benchmark preprocessing.

This pipeline keeps the benchmark practical for the local positives-only dataset
while preserving the core CCLMoff sequence representation:
    - keep bulge markers ('-') in both guide and target sequences
    - convert DNA bases T -> U for RNA-FM tokenization
    - create guide-level, load-balanced folds from positives first
    - generate fold-local synthetic negatives after splitting
"""

import argparse
import json
import math
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ARTIFACT_ROOT = os.path.join(SCRIPT_DIR, "artifacts", "cclmoff")
DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "offtarget_filtered.csv")
OUTPUT_DIR = os.path.join(ARTIFACT_ROOT, "preprocessed")

VALID_SEQ_PATTERN = re.compile(r"^[ATGCUN\-]+$")
CORE_COLUMNS = ["Guide_sequence", "Target_sequence", "label"]

VOCAB = OrderedDict(
    [
        ("[PAD]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("[MASK]", 3),
        ("A", 4),
        ("U", 5),
        ("G", 6),
        ("C", 7),
        ("N", 8),
        ("-", 9),
    ]
)


# ============================================================================
# Cleaning
# ============================================================================


def normalize_sequence(series):
    """Upper-case, trim, and convert DNA T bases to pseudo-RNA U bases."""
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("T", "U", regex=False)
    )


def load_and_clean(csv_path):
    """Load the positive-only benchmark dataset and preserve bulges."""
    print(f"Loading positives from {csv_path}...")
    raw_df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    print(f"  Loaded {len(raw_df):,} rows and {len(raw_df.columns)} columns")

    df = raw_df.copy()
    initial_rows = len(df)

    missing_mask = df["Guide_sequence"].isna() | df["Target_sequence"].isna()
    df = df.loc[~missing_mask].copy()

    df["Guide_sequence"] = df["Guide_sequence"].astype(str).str.strip().str.upper()
    df["Target_sequence"] = df["Target_sequence"].astype(str).str.strip().str.upper()

    empty_mask = (df["Guide_sequence"].str.len() == 0) | (df["Target_sequence"].str.len() == 0)
    df = df.loc[~empty_mask].copy()

    valid_mask = (
        df["Guide_sequence"].apply(lambda seq: bool(VALID_SEQ_PATTERN.fullmatch(seq)))
        & df["Target_sequence"].apply(lambda seq: bool(VALID_SEQ_PATTERN.fullmatch(seq)))
    )
    invalid_rows = df.loc[~valid_mask, ["Guide_sequence", "Target_sequence"]].head(5)
    df = df.loc[valid_mask].copy()

    guide_bulge_mask = df["Guide_sequence"].str.contains("-", regex=False)
    target_bulge_mask = df["Target_sequence"].str.contains("-", regex=False)

    df["Guide_sequence"] = normalize_sequence(df["Guide_sequence"])
    df["Target_sequence"] = normalize_sequence(df["Target_sequence"])

    df["guide_len"] = df["Guide_sequence"].str.len()
    df["target_len"] = df["Target_sequence"].str.len()
    df["target_has_gap"] = df["Target_sequence"].str.contains("-", regex=False)

    stats = {
        "input_rows": int(initial_rows),
        "rows_after_missing_filter": int(initial_rows - missing_mask.sum()),
        "rows_dropped_missing": int(missing_mask.sum()),
        "rows_dropped_empty": int(empty_mask.sum()),
        "rows_dropped_invalid": int((~valid_mask).sum()),
        "rows_after_cleaning": int(len(df)),
        "guide_bulge_rows_kept": int(guide_bulge_mask.sum()),
        "target_bulge_rows_kept": int(target_bulge_mask.sum()),
        "rows_with_any_bulge_kept": int((guide_bulge_mask | target_bulge_mask).sum()),
        "guide_length_range": [int(df["guide_len"].min()), int(df["guide_len"].max())],
        "target_length_range": [int(df["target_len"].min()), int(df["target_len"].max())],
        "invalid_row_examples": invalid_rows.to_dict(orient="records"),
    }

    print(f"  Dropped missing rows: {stats['rows_dropped_missing']:,}")
    print(f"  Dropped empty rows: {stats['rows_dropped_empty']:,}")
    print(f"  Dropped invalid rows: {stats['rows_dropped_invalid']:,}")
    print(f"  Kept guide bulge rows: {stats['guide_bulge_rows_kept']:,}")
    print(f"  Kept target bulge rows: {stats['target_bulge_rows_kept']:,}")
    print(
        "  Guide length range: "
        f"{stats['guide_length_range'][0]}-{stats['guide_length_range'][1]}"
    )
    print(
        "  Target length range: "
        f"{stats['target_length_range'][0]}-{stats['target_length_range'][1]}"
    )

    return df[["Guide_sequence", "Target_sequence", "guide_len", "target_len", "target_has_gap"]], stats


# ============================================================================
# Fold construction
# ============================================================================


def build_balanced_guide_folds(df, n_folds=5, seed=42):
    """Assign guides to folds with greedy load balancing on positive row counts."""
    guide_counts = df.groupby("Guide_sequence").size().to_dict()
    rng = np.random.default_rng(seed)

    grouped_guides = defaultdict(list)
    for guide, count in guide_counts.items():
        grouped_guides[count].append(guide)

    ordered_guides = []
    for count in sorted(grouped_guides.keys(), reverse=True):
        same_count_guides = sorted(grouped_guides[count])
        rng.shuffle(same_count_guides)
        ordered_guides.extend((guide, count) for guide in same_count_guides)

    fold_guides = [[] for _ in range(n_folds)]
    fold_row_counts = [0 for _ in range(n_folds)]

    for guide, count in ordered_guides:
        target_fold = min(
            range(n_folds),
            key=lambda idx: (fold_row_counts[idx], len(fold_guides[idx]), idx),
        )
        fold_guides[target_fold].append(guide)
        fold_row_counts[target_fold] += count

    folds = []
    for fold_idx, test_guides in enumerate(fold_guides):
        test_guides = sorted(test_guides)
        test_guide_set = set(test_guides)
        train_guides = sorted(set(guide_counts) - test_guide_set)

        test_df = df[df["Guide_sequence"].isin(test_guide_set)].copy()
        train_df = df[df["Guide_sequence"].isin(train_guides)].copy()

        folds.append(
            {
                "fold_idx": fold_idx,
                "train_guides": train_guides,
                "test_guides": test_guides,
                "train_df": train_df,
                "test_df": test_df,
                "train_positive_rows": int(len(train_df)),
                "test_positive_rows": int(len(test_df)),
            }
        )

    return folds


# ============================================================================
# Negative generation
# ============================================================================


def build_target_pools(df):
    """Create unique target pools keyed by (length, gap-status) signature."""
    target_by_signature = (
        df[["Target_sequence", "target_len", "target_has_gap"]]
        .drop_duplicates()
        .assign(target_signature=lambda x: list(zip(x["target_len"], x["target_has_gap"])))
        .groupby("target_signature")["Target_sequence"]
        .apply(list)
        .to_dict()
    )
    unique_targets = df["Target_sequence"].drop_duplicates().tolist()
    return {"by_signature": target_by_signature, "all_targets": unique_targets}


def sample_unique_targets(rng, pool, count):
    """Sample up to count unique targets without replacement."""
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        sampled = list(pool)
        rng.shuffle(sampled)
        return sampled
    indices = rng.choice(len(pool), size=count, replace=False)
    return [pool[idx] for idx in indices]


def generate_split_negatives(split_df, neg_ratio=1.0, random_state=42, fallback_pools=None):
    """
    Generate synthetic negatives inside a split.

    Each guide is paired with targets that appear elsewhere in the same split,
    preferring target sequences with the same length and bulge status as the
    guide's positive examples.
    """
    rng = np.random.default_rng(random_state)
    split_df = split_df.copy()
    split_df["target_signature"] = list(zip(split_df["target_len"], split_df["target_has_gap"]))

    split_pools = build_target_pools(split_df)
    fallback_pools = fallback_pools or split_pools

    split_target_lists = split_pools["by_signature"]
    split_targets = split_pools["all_targets"]
    fallback_target_lists = fallback_pools["by_signature"]
    fallback_targets = fallback_pools["all_targets"]
    guide_positive_targets = (
        split_df.groupby("Guide_sequence")["Target_sequence"].apply(lambda x: set(x.tolist())).to_dict()
    )

    negative_rows = []
    replacements_used = 0
    fallback_negatives = 0
    guides_with_replacements = defaultdict(int)
    guides_with_fallback = defaultdict(int)
    guide_negative_counts = {}

    for guide, guide_df in split_df.groupby("Guide_sequence", sort=False):
        positive_targets = guide_positive_targets[guide]
        primary_all = [target for target in split_targets if target not in positive_targets]
        fallback_all = [target for target in fallback_targets if target not in positive_targets]
        if not primary_all and not fallback_all:
            raise ValueError(f"No negative candidates available for guide {guide}")

        signatures = guide_df["target_signature"].tolist()
        target_count = int(math.ceil(len(guide_df) * float(neg_ratio)))
        expanded_signatures = [signatures[idx % len(signatures)] for idx in range(target_count)]

        signature_counts = defaultdict(int)
        for signature in expanded_signatures:
            signature_counts[signature] += 1

        used_targets = set()
        sampled_targets = []

        for signature, needed in signature_counts.items():
            preferred_pool = [
                target
                for target in split_target_lists.get(signature, [])
                if target not in positive_targets
            ]
            available_preferred = [target for target in preferred_pool if target not in used_targets]
            chosen = sample_unique_targets(rng, available_preferred, needed)
            sampled_targets.extend(chosen)
            used_targets.update(chosen)

            remaining = needed - len(chosen)
            if remaining <= 0:
                continue

            available_all = [target for target in primary_all if target not in used_targets]
            primary_fallback = sample_unique_targets(rng, available_all, remaining)
            sampled_targets.extend(primary_fallback)
            used_targets.update(primary_fallback)
            remaining -= len(primary_fallback)

            if remaining > 0:
                fallback_preferred_pool = [
                    target
                    for target in fallback_target_lists.get(signature, [])
                    if target not in positive_targets and target not in used_targets
                ]
                fallback_preferred = sample_unique_targets(rng, fallback_preferred_pool, remaining)
                sampled_targets.extend(fallback_preferred)
                used_targets.update(fallback_preferred)
                fallback_negatives += len(fallback_preferred)
                if fallback_preferred:
                    guides_with_fallback[guide] += len(fallback_preferred)
                remaining -= len(fallback_preferred)

            if remaining > 0:
                available_global = [target for target in fallback_all if target not in used_targets]
                fallback_global = sample_unique_targets(rng, available_global, remaining)
                sampled_targets.extend(fallback_global)
                used_targets.update(fallback_global)
                fallback_negatives += len(fallback_global)
                if fallback_global:
                    guides_with_fallback[guide] += len(fallback_global)
                remaining -= len(fallback_global)

            if remaining > 0:
                replacement_pool = preferred_pool or primary_all or fallback_all
                if not replacement_pool:
                    raise ValueError(f"Replacement pool is empty for guide {guide}")
                replacement_targets = rng.choice(replacement_pool, size=remaining, replace=True).tolist()
                sampled_targets.extend(replacement_targets)
                replacements_used += remaining
                guides_with_replacements[guide] += remaining

        if len(sampled_targets) != target_count:
            raise RuntimeError(
                f"Negative generation failed for guide {guide}: "
                f"expected {target_count}, got {len(sampled_targets)}"
            )

        guide_negative_counts[guide] = int(len(sampled_targets))
        negative_rows.extend(
            {
                "Guide_sequence": guide,
                "Target_sequence": target,
                "label": 0,
            }
            for target in sampled_targets
        )

    neg_df = pd.DataFrame(negative_rows)
    positive_pairs = set(
        split_df[["Guide_sequence", "Target_sequence"]].drop_duplicates().itertuples(index=False, name=None)
    )
    overlap_count = int(
        neg_df.merge(
            split_df[["Guide_sequence", "Target_sequence"]].drop_duplicates(),
            on=["Guide_sequence", "Target_sequence"],
            how="inner",
        ).shape[0]
    )
    if overlap_count != 0:
        raise RuntimeError(f"Negative generation produced {overlap_count} positive overlaps")

    negative_duplicate_rows = int(neg_df.duplicated(subset=["Guide_sequence", "Target_sequence"]).sum())
    neg_df = neg_df[CORE_COLUMNS]

    stats = {
        "positive_rows": int(len(split_df)),
        "negative_rows": int(len(neg_df)),
        "negative_overlap_with_positives": overlap_count,
        "negative_duplicate_rows": negative_duplicate_rows,
        "replacement_negatives": int(replacements_used),
        "fallback_negatives": int(fallback_negatives),
        "guides_with_replacements": {guide: int(count) for guide, count in guides_with_replacements.items()},
        "guides_with_fallback": {guide: int(count) for guide, count in guides_with_fallback.items()},
        "per_guide_negative_counts": {guide: int(count) for guide, count in guide_negative_counts.items()},
        "per_guide_positive_counts": {
            guide: int(count) for guide, count in split_df["Guide_sequence"].value_counts().to_dict().items()
        },
        "positive_pair_count": int(len(positive_pairs)),
        "negative_pair_count": int(
            len(neg_df[["Guide_sequence", "Target_sequence"]].drop_duplicates())
        ),
    }
    return neg_df, stats


def combine_and_shuffle_split(split_df, neg_ratio, seed, fallback_pools):
    """Attach fold-local negatives and return a benchmark-ready split."""
    pos_df = split_df[["Guide_sequence", "Target_sequence"]].copy()
    pos_df["label"] = 1

    neg_df, neg_stats = generate_split_negatives(
        split_df,
        neg_ratio=neg_ratio,
        random_state=seed,
        fallback_pools=fallback_pools,
    )
    combined = pd.concat([pos_df[CORE_COLUMNS], neg_df[CORE_COLUMNS]], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return combined, neg_stats


# ============================================================================
# Output helpers
# ============================================================================


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_length_distribution(df):
    summary = {}
    for column in ["Guide_sequence", "Target_sequence"]:
        lengths = df[column].str.len().value_counts().sort_index().to_dict()
        summary[column] = {str(length): int(count) for length, count in lengths.items()}
    return summary


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="CCLMoff benchmark preprocessing")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=1.0,
        help="Negative-to-positive ratio allocated per guide inside each split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CCLMoff Moderate Benchmark Preprocessing")
    print("=" * 70)

    cleaned_df, cleaning_stats = load_and_clean(args.input)

    print("\n[Step 2] Building guide-balanced folds...")
    folds = build_balanced_guide_folds(cleaned_df, n_folds=args.n_folds, seed=args.seed)
    fallback_pools = build_target_pools(cleaned_df)

    manifest = {
        "input": os.path.abspath(args.input),
        "output_dir": os.path.abspath(args.output_dir),
        "n_folds": int(args.n_folds),
        "neg_ratio": float(args.neg_ratio),
        "seed": int(args.seed),
        "cleaning": cleaning_stats,
        "sequence_length_distribution": summarize_length_distribution(cleaned_df),
        "folds": [],
    }

    print("\n[Step 3] Generating fold-local negatives and saving splits...")
    for fold in folds:
        fold_idx = fold["fold_idx"]
        fold_seed = args.seed + fold_idx

        train_ready, train_neg_stats = combine_and_shuffle_split(
            fold["train_df"],
            neg_ratio=args.neg_ratio,
            seed=fold_seed,
            fallback_pools=fallback_pools,
        )
        test_ready, test_neg_stats = combine_and_shuffle_split(
            fold["test_df"],
            neg_ratio=args.neg_ratio,
            seed=fold_seed + 10_000,
            fallback_pools=fallback_pools,
        )

        train_path = os.path.join(args.output_dir, f"fold_{fold_idx}_train.csv")
        test_path = os.path.join(args.output_dir, f"fold_{fold_idx}_test.csv")
        train_ready.to_csv(train_path, index=False)
        test_ready.to_csv(test_path, index=False)

        fold_manifest = {
            "fold": int(fold_idx),
            "train_guides": fold["train_guides"],
            "test_guides": fold["test_guides"],
            "guide_overlap": int(len(set(fold["train_guides"]) & set(fold["test_guides"]))),
            "train_positive_rows": int(fold["train_positive_rows"]),
            "test_positive_rows": int(fold["test_positive_rows"]),
            "train_total_rows": int(len(train_ready)),
            "test_total_rows": int(len(test_ready)),
            "train_stats": train_neg_stats,
            "test_stats": test_neg_stats,
            "train_csv": os.path.basename(train_path),
            "test_csv": os.path.basename(test_path),
        }
        manifest["folds"].append(fold_manifest)

        print(
            f"  Fold {fold_idx}: "
            f"train_pos={fold['train_positive_rows']:,}, train_total={len(train_ready):,} | "
            f"test_pos={fold['test_positive_rows']:,}, test_total={len(test_ready):,}"
        )

    manifest_path = os.path.join(args.output_dir, "preprocessing_manifest.json")
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    cleaned_path = os.path.join(args.output_dir, "cleaned_positives.csv")

    cleaned_df[["Guide_sequence", "Target_sequence"]].to_csv(cleaned_path, index=False)
    save_json(manifest_path, manifest)
    save_json(vocab_path, VOCAB)

    print("\n[Step 4] Saved artifacts")
    print(f"  Cleaned positives: {cleaned_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Vocab: {vocab_path}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
