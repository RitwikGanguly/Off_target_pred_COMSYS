#!/usr/bin/env python3
"""
CRISPRDeepOff validation using only foundation embeddings.

This script validates `model/new_model_MAIN/best_full.pt` on a sampled set of:
  - 600 off-target edges (label=1)
  - 400 on-target/non-off-target edges (label=0)

Embeddings:
  - Guide nodes: RNA-FM (foundation model)
  - Site nodes: DNABERT-2 (foundation model)
  - Edge features: handcrafted 25-dim (same format as training)

No k-mer embedding is used anywhere in this file.

Outputs are saved in:
  results/validation_crisprdeepoff/

Run:
  conda run -n rg_base python model_validation_crisprdeepoff.py
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import types
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offtarget_model import (
    HeteroGNN_TransformerConv,
    align_sequences,
    compute_cfd_like_score,
    compute_gc_content,
    compute_melting_temperature,
    compute_mismatch_vector,
    compute_weighted_mismatch_score,
    prepare_edge_attr_dict,
    prepare_edge_index_dict,
)


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = "/home/bernadettem/TNBC/bgnmf_benchmarking/chemi/off_target1"
DEFAULT_CHECKPOINT_PATH = os.path.join(BASE_DIR, "model/new_model_MAIN/best_full.pt")
DEFAULT_VALIDATION_CSV_PATH = os.path.join(
    BASE_DIR, "data/crisprDipOff_data/all_off_target.csv"
)
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results/validation_crisprdeepoff")

NODE_FEAT_DIM = 136
EMBED_DIM = 128
EDGE_FEAT_DIM = 25

EDGE_TYPE = ("guide", "targets", "site")
REV_EDGE_TYPE = ("site", "rev_targets", "guide")

PLOT_STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


# ============================================================================
# UTILS
# ============================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sanitize_seq(seq):
    seq = str(seq).upper()
    allowed = set("ACGTUN")
    return "".join(ch if ch in allowed else "N" for ch in seq)


def compute_stability(seq):
    seq = sanitize_seq(seq)
    gc_content = compute_gc_content(seq)
    stable_pairs = 0
    for i in range(max(len(seq) - 1, 0)):
        if seq[i : i + 2] in {"GC", "CG", "GG", "CC"}:
            stable_pairs += 1
    return (gc_content * 0.7) + ((stable_pairs / max(len(seq) - 1, 1)) * 0.3)


def expand_bio_features(gc, stab):
    return np.array(
        [
            gc,
            stab,
            gc**2,
            stab**2,
            gc * stab,
            abs(gc - 0.5),
            np.log(gc + 1e-6),
            np.log(stab + 1e-6),
        ],
        dtype=np.float32,
    )


def reduce_or_pad_embeddings(array_2d, target_dim, seed):
    arr = np.asarray(array_2d, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_samples, n_features = arr.shape
    if n_features == target_dim:
        return arr

    if n_features > target_dim:
        if n_samples >= 2:
            n_components = min(target_dim, n_features, n_samples)
            if n_components >= 2:
                pca = PCA(n_components=n_components, random_state=seed)
                reduced = pca.fit_transform(arr)
            else:
                reduced = arr[:, :n_components]
        else:
            reduced = arr[:, :target_dim]
    else:
        reduced = arr

    current_dim = reduced.shape[1]
    if current_dim < target_dim:
        pad = np.zeros((reduced.shape[0], target_dim - current_dim), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)

    return reduced.astype(np.float32)


def compute_single_handcrafted_edge_feature(guide_seq, site_seq):
    mv, total_mm = compute_mismatch_vector(guide_seq, site_seq)
    if len(mv) < 20:
        mv = mv + [0] * (20 - len(mv))
    else:
        mv = mv[:20]
    mm_norm = (
        total_mm / max(len(guide_seq), len(site_seq))
        if max(len(guide_seq), len(site_seq)) > 0
        else 0
    )
    align = align_sequences(guide_seq, site_seq)
    wt_mm = compute_weighted_mismatch_score(guide_seq, site_seq)
    cfd = compute_cfd_like_score(guide_seq, site_seq)
    tm = compute_melting_temperature(guide_seq, site_seq) / 100.0
    return mv + [mm_norm, align, wt_mm, cfd, tm]


def compute_all_metrics(y_true, y_scores, threshold=None):
    result = {}
    try:
        result["auroc"] = float(roc_auc_score(y_true, y_scores))
    except Exception:
        result["auroc"] = 0.0
    try:
        result["auprc"] = float(average_precision_score(y_true, y_scores))
    except Exception:
        result["auprc"] = 0.0

    if threshold is None:
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        best_idx = int(np.argmax(f1s))
        threshold = float(thresh[best_idx])
        result["optimal_f1"] = float(f1s[best_idx])
    else:
        result["optimal_f1"] = None

    result["threshold"] = float(threshold)
    y_binary = (y_scores >= threshold).astype(int)
    result["f1"] = float(f1_score(y_true, y_binary, zero_division=0))
    result["precision"] = float(precision_score(y_true, y_binary, zero_division=0))
    result["recall"] = float(recall_score(y_true, y_binary, zero_division=0))
    result["accuracy"] = float(accuracy_score(y_true, y_binary))
    result["mcc"] = float(matthews_corrcoef(y_true, y_binary))
    tn, fp, fn, tp = confusion_matrix(y_true, y_binary).ravel()
    result["tp"], result["fp"] = int(tp), int(fp)
    result["tn"], result["fn"] = int(tn), int(fn)
    result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # fixed-threshold diagnostics at 0.5 for sanity
    y_bin_05 = (y_scores >= 0.5).astype(int)
    result["f1_at_0p5"] = float(f1_score(y_true, y_bin_05, zero_division=0))
    result["precision_at_0p5"] = float(precision_score(y_true, y_bin_05, zero_division=0))
    result["recall_at_0p5"] = float(recall_score(y_true, y_bin_05, zero_division=0))
    result["accuracy_at_0p5"] = float(accuracy_score(y_true, y_bin_05))
    tn05, fp05, fn05, tp05 = confusion_matrix(y_true, y_bin_05).ravel()
    result["specificity_at_0p5"] = float(tn05 / (tn05 + fp05)) if (tn05 + fp05) > 0 else 0.0

    return result


# ============================================================================
# FOUNDATION LOADERS
# ============================================================================


class FoundationBundle:
    def __init__(self):
        self.rna_model = None
        self.rna_tokenizer = None
        self.dna_model = None
        self.dna_tokenizer = None
        self.rna_loaded = False
        self.dna_loaded = False
        self.rna_load_details = ""
        self.dna_load_details = ""


def _load_dnabert_local_backbone(snapshot_dir):
    package_name = "dnabert2_local_runtime"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [snapshot_dir]
    sys.modules[package_name] = pkg

    def load_submodule(name):
        fp = os.path.join(snapshot_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", fp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    config_mod = load_submodule("configuration_bert")
    load_submodule("bert_padding")
    try:
        load_submodule("flash_attn_triton")
    except Exception:
        pass
    bert_mod = load_submodule("bert_layers")

    config = config_mod.BertConfig.from_pretrained(snapshot_dir)
    # Force non-Triton path in bert_layers.py:
    # BertUnpadSelfAttention uses Triton only when p_dropout==0 and Triton exists.
    # Setting this >0 keeps inference on stable PyTorch attention path.
    config.attention_probs_dropout_prob = 0.1
    model = bert_mod.BertModel(config)

    state_dict = torch.load(os.path.join(snapshot_dir, "pytorch_model.bin"), map_location="cpu")
    state_dict = {k.replace("bert.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model, missing, unexpected


def load_foundation_models(device):
    bundle = FoundationBundle()
    print("\nLoading foundation models (strict foundation-only mode)...")

    # RNA-FM: use local safetensor snapshot and manual backbone mapping
    try:
        from multimolecule import RnaFmConfig, RnaFmModel, RnaTokenizer
        from safetensors.torch import load_file

        rna_snapshot = os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub/models--multimolecule--rnafm/snapshots/bcf7cc6e8f4385449c0288b3e1f41f21d68f1be3",
        )

        bundle.rna_tokenizer = RnaTokenizer.from_pretrained(rna_snapshot)
        rna_cfg = RnaFmConfig.from_pretrained(rna_snapshot)
        rna_model = RnaFmModel(rna_cfg)
        rna_sd = load_file(os.path.join(rna_snapshot, "model.safetensors"))
        backbone_sd = {k[len("model.") :]: v for k, v in rna_sd.items() if k.startswith("model.")}
        missing, unexpected = rna_model.load_state_dict(backbone_sd, strict=False)
        rna_model.eval().to(device)

        bundle.rna_model = rna_model
        bundle.rna_loaded = True
        bundle.rna_load_details = (
            f"local_backbone_load missing={len(missing)} unexpected={len(unexpected)}"
        )
        print(f"  RNA-FM: loaded ({bundle.rna_load_details})")
    except Exception as e:
        raise RuntimeError(f"RNA-FM load failed in foundation-only mode: {e}")

    # DNABERT-2: load local custom class and map weights
    try:
        from transformers import AutoTokenizer

        dna_snapshot = os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0",
        )

        bundle.dna_tokenizer = AutoTokenizer.from_pretrained(
            dna_snapshot, trust_remote_code=True
        )
        dna_model, missing, unexpected = _load_dnabert_local_backbone(dna_snapshot)
        dna_model.eval().to(device)

        bundle.dna_model = dna_model
        bundle.dna_loaded = True
        bundle.dna_load_details = (
            f"local_backbone_load missing={len(missing)} unexpected={len(unexpected)}"
        )
        print(f"  DNABERT-2: loaded ({bundle.dna_load_details})")
    except Exception as e:
        raise RuntimeError(f"DNABERT-2 load failed in foundation-only mode: {e}")

    return bundle


def batched_model_embeddings(
    sequences,
    tokenizer,
    model,
    device,
    batch_size,
    transform_fn=None,
    desc="embeddings",
):
    all_out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=desc):
            batch = sequences[i : i + batch_size]
            if transform_fn is not None:
                batch = [transform_fn(x) for x in batch]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if isinstance(outputs, tuple):
                hidden = outputs[0]
            elif hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs[0]

            pooled = hidden.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            all_out.append(pooled)

    return np.concatenate(all_out, axis=0)


def build_foundation_node_features(guide_seqs, site_seqs, bundle, device, batch_size, seed):
    guide_raw = batched_model_embeddings(
        sequences=[sanitize_seq(s) for s in guide_seqs],
        tokenizer=bundle.rna_tokenizer,
        model=bundle.rna_model,
        device=device,
        batch_size=batch_size,
        transform_fn=lambda s: s.replace("T", "U"),
        desc="Guide RNA-FM embeddings",
    )

    site_raw = batched_model_embeddings(
        sequences=[sanitize_seq(s) for s in site_seqs],
        tokenizer=bundle.dna_tokenizer,
        model=bundle.dna_model,
        device=device,
        batch_size=batch_size,
        transform_fn=None,
        desc="Site DNABERT-2 embeddings",
    )

    guide_emb = reduce_or_pad_embeddings(guide_raw, EMBED_DIM, seed)
    site_emb = reduce_or_pad_embeddings(site_raw, EMBED_DIM, seed)

    guide_bio = np.array(
        [
            expand_bio_features(
                compute_gc_content(sanitize_seq(s)),
                compute_stability(sanitize_seq(s)),
            )
            for s in guide_seqs
        ],
        dtype=np.float32,
    )
    site_bio = np.array(
        [
            expand_bio_features(
                compute_gc_content(sanitize_seq(s)),
                compute_stability(sanitize_seq(s)),
            )
            for s in site_seqs
        ],
        dtype=np.float32,
    )

    guide_full = np.concatenate([guide_emb, guide_bio], axis=1)
    site_full = np.concatenate([site_emb, site_bio], axis=1)

    scaler_g = StandardScaler()
    scaler_s = StandardScaler()
    guide_std = scaler_g.fit_transform(guide_full).astype(np.float32)
    site_std = scaler_s.fit_transform(site_full).astype(np.float32)

    meta = {
        "guide_source": "RNA-FM",
        "site_source": "DNABERT-2",
        "guide_raw_dim": int(guide_raw.shape[1]),
        "site_raw_dim": int(site_raw.shape[1]),
        "guide_post_dim": int(guide_emb.shape[1]),
        "site_post_dim": int(site_emb.shape[1]),
        "rna_load_details": bundle.rna_load_details,
        "dna_load_details": bundle.dna_load_details,
    }
    return guide_std, site_std, meta


def build_handcrafted_edge_features(sampled_df, labels, guide_to_idx, site_to_idx):
    edge_index_list = []
    edge_features_list = []
    edge_labels_list = []

    for i in tqdm(range(len(sampled_df)), desc="Edge handcrafted features"):
        row = sampled_df.iloc[i]
        gs = sanitize_seq(row["Target sgRNA"])
        ss = sanitize_seq(row["Off Target sgRNA"])
        gi = guide_to_idx[gs]
        si = site_to_idx[ss]

        ef = compute_single_handcrafted_edge_feature(gs, ss)
        edge_index_list.append([gi, si])
        edge_features_list.append(ef)
        edge_labels_list.append(labels[i])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    edge_label = torch.tensor(edge_labels_list, dtype=torch.float)
    return edge_index, edge_attr, edge_label


def build_heterodata(guide_x, site_x, edge_index, edge_attr, edge_label):
    data = HeteroData()
    data["guide"].x = torch.tensor(guide_x, dtype=torch.float)
    data["site"].x = torch.tensor(site_x, dtype=torch.float)
    data["guide"].num_nodes = guide_x.shape[0]
    data["site"].num_nodes = site_x.shape[0]

    data[EDGE_TYPE].edge_index = edge_index
    data[EDGE_TYPE].edge_attr = edge_attr
    data[EDGE_TYPE].edge_label = edge_label

    data[REV_EDGE_TYPE].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    data[REV_EDGE_TYPE].edge_attr = edge_attr.clone()

    data[EDGE_TYPE].edge_label_index = edge_index
    data[EDGE_TYPE].edge_label_attr = edge_attr
    return data


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hparams"]

    model = HeteroGNN_TransformerConv(
        guide_in_channels=NODE_FEAT_DIM,
        site_in_channels=NODE_FEAT_DIM,
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        edge_feat_dim=hp["edge_feat_dim"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        dropout=hp["dropout"],
        edge_dropout=hp["edge_dropout"],
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model, ckpt, hp, missing, unexpected


def run_inference(model, data, device):
    model.eval()
    data_dev = data.to(device)
    edge_label_index = data_dev[EDGE_TYPE].edge_label_index
    edge_features = data_dev[EDGE_TYPE].edge_label_attr

    with torch.no_grad():
        edge_index_dict = prepare_edge_index_dict(data_dev)
        edge_attr_dict = prepare_edge_attr_dict(data_dev)
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

        h_dict = model.encode(data_dev.x_dict, edge_index_dict, edge_attr_dict)
        logits = model.decode(h_dict, edge_label_index, edge_features)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    y_true = data_dev[EDGE_TYPE].edge_label.detach().cpu().numpy().reshape(-1)
    return y_true, probs


def save_plots(y_true, probs, metrics, output_dir, prefix, title_suffix):
    plt.rcParams.update(PLOT_STYLE)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#dc2626", linewidth=2.5, label=f"AUROC = {metrics['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.12, color="#dc2626")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title_suffix}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.pdf")
    plt.savefig(roc_path)
    plt.close()

    # PR
    prec_arr, rec_arr, thresh_pr = precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_arr, prec_arr, color="#16a34a", linewidth=2.5, label=f"AUPRC = {metrics['auprc']:.4f}")
    ax.fill_between(rec_arr, prec_arr, alpha=0.12, color="#16a34a")
    if len(thresh_pr) > 0:
        f1s = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
        best_i = int(np.argmax(f1s))
        ax.scatter(
            [rec_arr[best_i]],
            [prec_arr[best_i]],
            color="#f59e0b",
            s=120,
            zorder=5,
            label=f"Best F1={f1s[best_i]:.4f} @ {thresh_pr[best_i]:.4f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {title_suffix}", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.pdf")
    plt.savefig(pr_path)
    plt.close()

    # score distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        probs[y_true == 1],
        bins=40,
        alpha=0.7,
        color="#dc2626",
        label=f"Off-target (n={int(y_true.sum())})",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].hist(
        probs[y_true == 0],
        bins=20,
        alpha=0.7,
        color="#2563eb",
        label=f"On-target/Not-OT (n={int((1-y_true).sum())})",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].axvline(
        metrics["threshold"],
        color="#f59e0b",
        linestyle="--",
        linewidth=2,
        label=f"Threshold {metrics['threshold']:.4f}",
    )
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    bp = axes[1].boxplot(
        [probs[y_true == 0].ravel(), probs[y_true == 1].ravel()],
        labels=["On-target/Not-OT", "Off-target"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="red", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("#3b82f6")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#ef4444")
    bp["boxes"][1].set_alpha(0.6)
    axes[1].axhline(metrics["threshold"], color="#f59e0b", linestyle="--", linewidth=2)
    axes[1].set_ylabel("Predicted Probability")
    axes[1].set_title("Score by Class", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    score_path = os.path.join(output_dir, f"{prefix}_score_distributions.pdf")
    plt.savefig(score_path)
    plt.close()

    # confusion matrix
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred On-target", "Pred Off-target"],
        yticklabels=["True On-target", "True Off-target"],
        linewidths=2,
        linecolor="white",
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {title_suffix}", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.pdf")
    plt.savefig(cm_path)
    plt.close()

    return {
        "roc_curve": roc_path,
        "pr_curve": pr_path,
        "score_distributions": score_path,
        "confusion_matrix": cm_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="CRISPRDeepOff validation (foundation-only)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--validation_csv", type=str, default=DEFAULT_VALIDATION_CSV_PATH)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n_sample_ot", type=int, default=450)
    parser.add_argument("--n_sample_not", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("CRISPRDEEPOFF VALIDATION (FOUNDATION-ONLY)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation CSV: {args.validation_csv}")
    print(f"Results dir: {args.results_dir}")

    # ------------------------------------------------------------------
    # Data sampling: exactly 450 positive + 200 negative
    # ------------------------------------------------------------------
    val_df = pd.read_csv(args.validation_csv)
    val_df["Target sgRNA"] = val_df["Target sgRNA"].astype(str).map(sanitize_seq)
    val_df["Off Target sgRNA"] = val_df["Off Target sgRNA"].astype(str).map(sanitize_seq)
    val_df["label"] = val_df["label"].astype(int)

    df_label1 = val_df[val_df["label"] == 1]
    df_label0 = val_df[val_df["label"] == 0]

    if len(df_label1) < args.n_sample_ot:
        raise ValueError(
            f"Requested {args.n_sample_ot} off-target edges but only {len(df_label1)} available"
        )
    if len(df_label0) < args.n_sample_not:
        raise ValueError(
            f"Requested {args.n_sample_not} on-target/non-off-target edges but only {len(df_label0)} available"
        )

    df_ot = df_label1.sample(n=args.n_sample_ot, random_state=args.seed).reset_index(drop=True)
    df_not = df_label0.sample(n=args.n_sample_not, random_state=args.seed).reset_index(drop=True)

    sampled_df = pd.concat([df_ot, df_not], ignore_index=True)
    labels = sampled_df["label"].values.astype(np.float32)

    perm = np.random.permutation(len(sampled_df))
    sampled_df = sampled_df.iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    print("\nSampling summary:")
    print(f"  total: {len(sampled_df):,}")
    print(f"  off-target (label=1): {int(labels.sum()):,}")
    print(f"  on-target/non-off-target (label=0): {int((1 - labels).sum()):,}")

    # ------------------------------------------------------------------
    # Node mapping
    # ------------------------------------------------------------------
    unique_guides = sampled_df["Target sgRNA"].unique()
    unique_sites = sampled_df["Off Target sgRNA"].unique()

    guide_to_idx = {seq: idx for idx, seq in enumerate(unique_guides)}
    site_to_idx = {seq: idx for idx, seq in enumerate(unique_sites)}

    print(f"  unique guides: {len(unique_guides):,}")
    print(f"  unique sites: {len(unique_sites):,}")

    # ------------------------------------------------------------------
    # Load model checkpoint
    # ------------------------------------------------------------------
    model_template, ckpt, hp, missing, unexpected = load_model(args.checkpoint, device)
    del model_template

    print("\nCheckpoint:")
    print(f"  epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  val_auroc: {float(ckpt.get('val_auroc', float('nan'))):.6f}")
    print(f"  val_auprc: {float(ckpt.get('val_auprc', float('nan'))):.6f}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    print(f"  hparams: {hp}")

    # ------------------------------------------------------------------
    # Foundation embeddings only
    # ------------------------------------------------------------------
    foundation_bundle = load_foundation_models(device)

    guide_x, site_x, node_meta = build_foundation_node_features(
        guide_seqs=unique_guides,
        site_seqs=unique_sites,
        bundle=foundation_bundle,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("\nNode embedding summary:")
    print(f"  guide features: {guide_x.shape}")
    print(f"  site features: {site_x.shape}")
    print(f"  guide source: {node_meta['guide_source']} (raw {node_meta['guide_raw_dim']} -> {node_meta['guide_post_dim']})")
    print(f"  site source: {node_meta['site_source']} (raw {node_meta['site_raw_dim']} -> {node_meta['site_post_dim']})")

    # ------------------------------------------------------------------
    # Edge features (training-compatible decoder input)
    # ------------------------------------------------------------------
    edge_index, edge_attr, edge_label = build_handcrafted_edge_features(
        sampled_df, labels, guide_to_idx, site_to_idx
    )

    print("\nEdge feature summary:")
    print(f"  edge_index: {tuple(edge_index.shape)}")
    print(f"  edge_attr: {tuple(edge_attr.shape)}")
    print(f"  positives: {int(edge_label.sum().item())}")
    print(f"  negatives: {int((1 - edge_label).sum().item())}")

    # ------------------------------------------------------------------
    # Build graph and run inference
    # ------------------------------------------------------------------
    data = build_heterodata(guide_x, site_x, edge_index, edge_attr, edge_label)

    model, _, _, missing_run, unexpected_run = load_model(args.checkpoint, device)
    if missing_run:
        print(f"  Missing keys (first 5): {missing_run[:5]}")
    if unexpected_run:
        print(f"  Unexpected keys (first 5): {unexpected_run[:5]}")

    y_true, probs = run_inference(model, data, device)
    metrics = compute_all_metrics(y_true, probs)

    print("\nValidation metrics:")
    for k in ["auroc", "auprc", "f1", "precision", "recall", "specificity", "mcc", "accuracy"]:
        print(f"  {k:>12s}: {metrics[k]:.4f}")
    print(f"  {'threshold':>12s}: {metrics['threshold']:.4f}")
    print(
        f"  TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}"
    )
    print(
        "\nSanity at threshold=0.5: "
        f"F1={metrics['f1_at_0p5']:.4f}, "
        f"Precision={metrics['precision_at_0p5']:.4f}, "
        f"Recall={metrics['recall_at_0p5']:.4f}, "
        f"Specificity={metrics['specificity_at_0p5']:.4f}, "
        f"Accuracy={metrics['accuracy_at_0p5']:.4f}"
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    prefix = "crisprdeepoff_foundation_only"
    plot_paths = save_plots(
        y_true=y_true,
        probs=probs,
        metrics=metrics,
        output_dir=args.results_dir,
        prefix=prefix,
        title_suffix="DeepCrispr Data Validation",
    )

    payload = {
        "experiment": {
            "id": "foundation_only",
            "node_mode": "foundation",
            "edge_mode": "handcrafted_training_compatible",
            "title": "DeepCrispr Data Validation",
        },
        "checkpoint": args.checkpoint,
        "checkpoint_meta": {
            "epoch": ckpt.get("epoch", None),
            "val_auroc": float(ckpt.get("val_auroc", 0.0)),
            "val_auprc": float(ckpt.get("val_auprc", 0.0)),
        },
        "sampling": {
            "seed": args.seed,
            "n_sample_ot": int(args.n_sample_ot),
            "n_sample_not": int(args.n_sample_not),
            "n_total": int(len(sampled_df)),
            "num_guides": int(len(unique_guides)),
            "num_sites": int(len(unique_sites)),
        },
        "node_embedding_meta": node_meta,
        "edge_embedding_meta": {
            "edge_source": "handcrafted_25d_training_compatible",
            "edge_dim": int(edge_attr.shape[1]),
        },
        "metrics": metrics,
        "label1_stats": {
            "n": int(y_true.sum()),
            "mean_score": float(probs[y_true == 1].mean()),
            "median_score": float(np.median(probs[y_true == 1])),
            "detected_at_threshold": int((probs[y_true == 1] >= metrics["threshold"]).sum()),
            "detected_at_0p5": int((probs[y_true == 1] >= 0.5).sum()),
        },
        "label0_stats": {
            "n": int((1 - y_true).sum()),
            "mean_score": float(probs[y_true == 0].mean()),
            "median_score": float(np.median(probs[y_true == 0])),
            "rejected_at_threshold": int((probs[y_true == 0] < metrics["threshold"]).sum()),
            "rejected_at_0p5": int((probs[y_true == 0] < 0.5).sum()),
        },
        "plot_paths": plot_paths,
    }

    metrics_json = os.path.join(args.results_dir, f"{prefix}_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics: {metrics_json}")

    pred_csv = os.path.join(args.results_dir, f"{prefix}_predictions.csv")
    pred_df = pd.DataFrame({"y_true": y_true.astype(int), "y_score": probs})
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions: {pred_csv}")

    # compact summary csv for quick comparison
    summary_csv = os.path.join(args.results_dir, "crisprdeepoff_foundation_only_summary.csv")
    pd.DataFrame(
        [
            {
                "experiment_id": "foundation_only",
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "mcc": metrics["mcc"],
                "accuracy": metrics["accuracy"],
                "threshold": metrics["threshold"],
                "f1_at_0p5": metrics["f1_at_0p5"],
                "precision_at_0p5": metrics["precision_at_0p5"],
                "recall_at_0p5": metrics["recall_at_0p5"],
                "specificity_at_0p5": metrics["specificity_at_0p5"],
                "accuracy_at_0p5": metrics["accuracy_at_0p5"],
                "metrics_json": metrics_json,
            }
        ]
    ).to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
