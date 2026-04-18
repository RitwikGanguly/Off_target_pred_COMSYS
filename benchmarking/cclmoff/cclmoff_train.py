#!/usr/bin/env python3
"""
CCLMoff training for the local moderate benchmark - DOWNGRADED VERSION.

Simplified training for baseline benchmarking:
    - Optimizer: AdamW
    - Head LR: 1e-3 (constant, no warmup)
    - Backbone: frozen (no LR needed)
    - Weight decay: 0.01
    - Batch size: 128
    - Epochs: 100
    - NO learning rate scheduling
    - Gradient clipping: 1.0
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised via runtime guard
    torch = None
    nn = None
    GradScaler = None
    autocast = None
    DataLoader = None
    TORCH_IMPORT_ERROR = exc

    class Dataset:  # pragma: no cover - import-time placeholder
        pass

else:
    TORCH_IMPORT_ERROR = None


# Local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from chemi.off_target1.benchmarking.cclmoff.crispr_off_T.cclmoff_model import CCLMoff


# ============================================================================
# Paths
# ============================================================================

ARTIFACT_ROOT = os.path.join(SCRIPT_DIR, "artifacts", "cclmoff")
PREPROC_DIR = os.path.join(ARTIFACT_ROOT, "preprocessed")
CKPT_DIR = os.path.join(ARTIFACT_ROOT, "checkpoints")
LOG_DIR = os.path.join(ARTIFACT_ROOT, "logs")
RESULTS_DIR = os.path.join(ARTIFACT_ROOT, "results")


# ============================================================================
# Helpers
# ============================================================================


def require_training_dependencies():
    if TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "cclmoff_train.py requires PyTorch. Install `torch` before running training."
        ) from TORCH_IMPORT_ERROR


def set_seed(seed):
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_log(log_file, message):
    print(message)
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def ensure_fold_files(preproc_dir, n_folds):
    missing = []
    for fold_idx in range(n_folds):
        for split in ("train", "test"):
            path = os.path.join(preproc_dir, f"fold_{fold_idx}_{split}.csv")
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing preprocessed fold files. Run cclmoff_preprocess.py first.\n"
            f"{missing_text}"
        )


def build_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)

    if labels.size == 0:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    binary = (preds >= 0.5).astype(int)
    label_classes = np.unique(labels)

    if label_classes.size >= 2:
        auroc = float(roc_auc_score(labels, preds))
        auprc = float(average_precision_score(labels, preds))
        balanced_accuracy = float(balanced_accuracy_score(labels, binary))
    else:
        auroc = 0.0
        auprc = 0.0
        balanced_accuracy = 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(labels, binary, zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(labels, binary, zero_division=0)),
        "recall": float(recall_score(labels, binary, zero_division=0)),
    }


class LinearWarmupConstantScheduler:
    """Epoch-level linear warmup followed by constant learning rate."""

    def __init__(self, optimizer, warmup_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    @staticmethod
    def warmup_factor(epoch, warmup_epochs):
        if warmup_epochs <= 0:
            return 1.0
        if epoch < 1:
            return 0.0
        if epoch <= warmup_epochs:
            return float(epoch) / float(warmup_epochs)
        return 1.0

    def step(self, epoch):
        factor = self.warmup_factor(epoch, self.warmup_epochs)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * factor
        return factor


# ============================================================================
# Dataset / DataLoader
# ============================================================================


class RNAFMDataset(Dataset):
    """Dataset returning CCLMoff-formatted guide<sep>target pairs."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        self.guides = df["Guide_sequence"].astype(str).values
        self.targets = df["Target_sequence"].astype(str).values
        self.labels = df["label"].astype(np.float32).values

    def __getitem__(self, index):
        seq = f"{self.guides[index]}<sep>{self.targets[index]}"
        seq = seq.replace("_", "-")
        return seq, float(self.labels[index])

    def __len__(self):
        return len(self.labels)


def collate_rnafm(batch, alphabet):
    seqs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    data = [(f"s{i}", seq) for i, seq in enumerate(seqs)]
    _, _, batch_tokens = alphabet.get_batch_converter()(data)
    return batch_tokens, labels


def create_loaders(train_csv, val_csv, alphabet, batch_size, num_workers, pin_memory):
    train_ds = RNAFMDataset(train_csv)
    val_ds = RNAFMDataset(val_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=lambda batch: collate_rnafm(batch, alphabet),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=lambda batch: collate_rnafm(batch, alphabet),
    )
    return train_loader, val_loader


# ============================================================================
# Training / Evaluation
# ============================================================================


def amp_context(use_amp):
    if use_amp:
        return autocast(device_type="cuda", enabled=True)
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_norm=1.0, use_amp=True):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for tokens, labels in tqdm(loader, total=len(loader), desc="  Training", leave=False):
        tokens = tokens.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with amp_context(use_amp):
            outputs = model(tokens)
        loss = criterion(outputs.float(), labels.float()) if use_amp else criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().squeeze(1).numpy().tolist())
        all_preds.extend(outputs.detach().float().cpu().squeeze(1).numpy().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, build_metrics(all_labels, all_preds)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def evaluate(model, loader, criterion, device, use_amp=True, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for tokens, labels in tqdm(loader, total=len(loader), desc="  Evaluating", leave=False):
        tokens = tokens.to(device)
        labels = labels.to(device).unsqueeze(1)

        with amp_context(use_amp):
            outputs = model(tokens)
        loss = criterion(outputs.float(), labels.float()) if use_amp else criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().squeeze(1).numpy().tolist())
        all_preds.extend(outputs.detach().float().cpu().squeeze(1).numpy().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    metrics = build_metrics(all_labels, all_preds)
    if return_preds:
        return avg_loss, metrics, all_labels, all_preds
    return avg_loss, metrics


def save_roc_curve(labels, preds, output_path, title):
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)

    if np.unique(labels).size < 2:
        return {
            "saved": False,
            "reason": "Only one class present in labels; ROC curve is undefined.",
            "roc_auc": None,
        }

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = float(roc_auc_score(labels, preds))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#6b7280", linewidth=1.5, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return {
        "saved": True,
        "reason": "",
        "roc_auc": roc_auc,
    }


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(path, model, optimizer, fold_idx, epoch, seed, metrics, config):
    state = {
        "fold": int(fold_idx),
        "epoch": int(epoch),
        "seed": int(seed),
        "config": config,
        "metrics": metrics,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


def format_epoch_message(epoch, epochs, train_loss, train_metrics, val_loss, val_metrics, lr, elapsed):
    return (
        f"  Epoch {epoch:2d}/{epochs} | "
        f"Train Loss={train_loss:.4f} AUROC={train_metrics['auroc']:.4f} "
        f"AUPRC={train_metrics['auprc']:.4f} | "
        f"Val Loss={val_loss:.4f} AUROC={val_metrics['auroc']:.4f} "
        f"AUPRC={val_metrics['auprc']:.4f} F1={val_metrics['f1']:.4f} "
        f"BalAcc={val_metrics['balanced_accuracy']:.4f} "
        f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f} | "
        f"LR={lr:.6f} | {elapsed:.1f}s"
    )


def train_fold(fold_idx, config, device, log_file, preproc_dir, ckpt_dir, results_dir):
    seed = config["seed"] + fold_idx
    set_seed(seed)

    train_csv = os.path.join(preproc_dir, f"fold_{fold_idx}_train.csv")
    test_csv = os.path.join(preproc_dir, f"fold_{fold_idx}_test.csv")

    model = CCLMoff()
    alphabet = model.get_alphabet()

    runtime_device = device
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        # If user pinned a specific GPU (e.g., cuda:1), keep training on that GPU only.
        if device.index is not None:
            write_log(
                log_file,
                f"  Fold {fold_idx}: pinned to single GPU cuda:{device.index} (DataParallel disabled)",
            )
        else:
            device_ids = list(range(torch.cuda.device_count()))
            runtime_device = torch.device("cuda", device_ids[0])
            write_log(log_file, f"  Fold {fold_idx}: using {len(device_ids)} GPUs")
            model = nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )

    model = model.to(runtime_device)
    pin_memory = device.type == "cuda"
    train_loader, val_loader = create_loaders(
        train_csv=train_csv,
        val_csv=test_csv,
        alphabet=alphabet,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    base_model = unwrap_model(model)
    backbone_params = []
    head_params = []
    for name, param in base_model.named_parameters():
        if "rna_model" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        head_params,
        lr=config["head_lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCELoss()
    use_amp = runtime_device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    write_log(log_file, "\n" + "=" * 68)
    write_log(log_file, f"Fold {fold_idx} | Seed {seed} | Device: {runtime_device}")
    write_log(
        log_file,
        f"Train rows: {len(train_loader.dataset):,} | Val rows: {len(val_loader.dataset):,}",
    )
    write_log(
        log_file,
        f"Backbone params (frozen): {sum(p.numel() for p in backbone_params):,} | "
        f"Head params: {sum(p.numel() for p in head_params):,}",
    )
    write_log(log_file, "=" * 68)

    history = []
    best_by_auprc = -1.0
    best_epoch = 0
    best_ckpt_path = os.path.join(ckpt_dir, f"best_by_auprc_fold_{fold_idx}.pt")
    final_ckpt_path = os.path.join(ckpt_dir, f"final_fold_{fold_idx}.pt")
    test_pred_path = os.path.join(results_dir, f"test_predictions_fold_{fold_idx}.csv")
    test_metrics_path = os.path.join(results_dir, f"test_metrics_fold_{fold_idx}.json")
    test_roc_curve_path = os.path.join(results_dir, f"test_roc_curve_fold_{fold_idx}.png")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        current_lr = config["head_lr"]

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=runtime_device,
            max_norm=config["max_norm"],
            use_amp=use_amp,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=runtime_device,
            use_amp=use_amp,
        )

        elapsed = time.time() - epoch_start
        record = {
            "fold": int(fold_idx),
            "seed": int(seed),
            "epoch": int(epoch),
            "head_lr": float(current_lr),
            "train_loss": float(train_loss),
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_f1": train_metrics["f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": float(val_loss),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "epoch_time_sec": float(elapsed),
        }
        history.append(record)

        write_log(
            log_file,
            format_epoch_message(
                epoch=epoch,
                epochs=config["epochs"],
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics,
                lr=current_lr,
                elapsed=elapsed,
            ),
        )

        if val_metrics["auprc"] > best_by_auprc:
            best_by_auprc = val_metrics["auprc"]
            best_epoch = epoch
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                fold_idx=fold_idx,
                epoch=epoch,
                seed=seed,
                metrics=record,
                config=config,
            )
            write_log(
                log_file,
                f"    Saved best-by-AUPRC checkpoint for fold {fold_idx} at epoch {epoch}",
            )

    final_record = history[-1]
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        fold_idx=fold_idx,
        epoch=config["epochs"],
        seed=seed,
        metrics=final_record,
        config=config,
    )
    write_log(
        log_file,
        f"Fold {fold_idx} complete. Best AUPRC={best_by_auprc:.4f} at epoch {best_epoch}",
    )

    # Test-time inference from best saved checkpoint
    best_ckpt = torch.load(best_ckpt_path, map_location=runtime_device)
    unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_loss, test_metrics, test_labels, test_preds = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=runtime_device,
        use_amp=use_amp,
        return_preds=True,
    )

    pred_df = pd.DataFrame(
        {
            "label": np.asarray(test_labels, dtype=np.float32),
            "prediction": np.asarray(test_preds, dtype=np.float32),
            "prediction_binary_0_5": (np.asarray(test_preds, dtype=np.float32) >= 0.5).astype(int),
        }
    )
    pred_df.to_csv(test_pred_path, index=False)

    roc_info = save_roc_curve(
        labels=test_labels,
        preds=test_preds,
        output_path=test_roc_curve_path,
        title=f"Fold {fold_idx} Test ROC Curve",
    )
    if not roc_info["saved"]:
        test_roc_curve_path = ""
        write_log(log_file, f"  Fold {fold_idx} test ROC skipped: {roc_info['reason']}")

    fold_test_payload = {
        "fold": int(fold_idx),
        "best_checkpoint": best_ckpt_path,
        "test_loss": float(test_loss),
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "roc_curve_saved": bool(roc_info["saved"]),
        "roc_curve_path": test_roc_curve_path,
        "test_predictions_path": test_pred_path,
    }
    with open(test_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(fold_test_payload, handle, indent=2)

    write_log(
        log_file,
        f"  Fold {fold_idx} test metrics | Loss={test_loss:.4f} AUROC={test_metrics['auroc']:.4f} "
        f"AUPRC={test_metrics['auprc']:.4f} F1={test_metrics['f1']:.4f}",
    )
    write_log(log_file, f"  Saved test predictions: {test_pred_path}")
    write_log(log_file, f"  Saved test metrics: {test_metrics_path}")
    if roc_info["saved"]:
        write_log(log_file, f"  Saved ROC curve: {test_roc_curve_path}")

    summary = {
        "fold": int(fold_idx),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_by_auprc),
        "best_checkpoint": best_ckpt_path,
        "final_checkpoint": final_ckpt_path,
        "final_val_loss": final_record["val_loss"],
        "final_val_auroc": final_record["val_auroc"],
        "final_val_auprc": final_record["val_auprc"],
        "final_val_f1": final_record["val_f1"],
        "final_val_balanced_accuracy": final_record["val_balanced_accuracy"],
        "final_val_precision": final_record["val_precision"],
        "final_val_recall": final_record["val_recall"],
        "test_metrics_path": test_metrics_path,
        "test_predictions_path": test_pred_path,
        "test_roc_curve_path": test_roc_curve_path,
        "test_loss": float(test_loss),
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
    }

    return history, summary


def summarize_cv(fold_summaries):
    metrics = [
        "best_val_auprc",
        "final_val_loss",
        "final_val_auroc",
        "final_val_auprc",
        "final_val_f1",
        "final_val_balanced_accuracy",
        "final_val_precision",
        "final_val_recall",
        "test_loss",
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
    ]
    summary = {
        "n_folds_completed": int(len(fold_summaries)),
        "folds": fold_summaries,
        "metrics": {},
    }
    fold_df = pd.DataFrame(fold_summaries)
    for metric in metrics:
        values = fold_df[metric].astype(float).values
        summary["metrics"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train the local moderate CCLMoff benchmark")
    parser.add_argument("--preproc_dir", type=str, default=PREPROC_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    require_training_dependencies()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    ensure_fold_files(args.preproc_dir, args.n_folds)

    config = {
        "head_lr": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "max_norm": float(args.max_norm),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
    }

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = os.path.join(args.log_dir, "training_log.txt")
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 70 + "\n")
        handle.write(f"Training started: {datetime.now().isoformat()}\n")
        handle.write(f"Config: {json.dumps(config, indent=2)}\n")
        handle.write("=" * 70 + "\n")

    write_log(log_file, "=" * 70)
    write_log(log_file, "CCLMoff Moderate Benchmark Training")
    write_log(log_file, "=" * 70)
    write_log(log_file, f"Device: {device}")
    write_log(log_file, f"Preprocessed data: {args.preproc_dir}")
    write_log(log_file, f"Checkpoints: {args.ckpt_dir}")
    write_log(log_file, f"Results: {args.results_dir}")

    all_history = []
    fold_summaries = []

    for fold_idx in range(args.n_folds):
        fold_history, fold_summary = train_fold(
            fold_idx=fold_idx,
            config=config,
            device=device,
            log_file=log_file,
            preproc_dir=args.preproc_dir,
            ckpt_dir=args.ckpt_dir,
            results_dir=args.results_dir,
        )
        all_history.extend(fold_history)
        fold_summaries.append(fold_summary)

    history_df = pd.DataFrame(all_history)
    fold_summary_df = pd.DataFrame(fold_summaries)
    cv_summary = summarize_cv(fold_summaries)

    history_path = os.path.join(args.results_dir, "training_history.csv")
    fold_summary_path = os.path.join(args.results_dir, "fold_summary.csv")
    cv_summary_path = os.path.join(args.results_dir, "cv_summary.json")

    history_df.to_csv(history_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)
    with open(cv_summary_path, "w", encoding="utf-8") as handle:
        json.dump(cv_summary, handle, indent=2)

    write_log(log_file, "\nSummary:")
    write_log(
        log_file,
        "  Final AUROC: "
        f"{cv_summary['metrics']['final_val_auroc']['mean']:.4f} ± "
        f"{cv_summary['metrics']['final_val_auroc']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Final AUPRC: "
        f"{cv_summary['metrics']['final_val_auprc']['mean']:.4f} ± "
        f"{cv_summary['metrics']['final_val_auprc']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Final F1: "
        f"{cv_summary['metrics']['final_val_f1']['mean']:.4f} ± "
        f"{cv_summary['metrics']['final_val_f1']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Final Balanced Accuracy: "
        f"{cv_summary['metrics']['final_val_balanced_accuracy']['mean']:.4f} ± "
        f"{cv_summary['metrics']['final_val_balanced_accuracy']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Test AUROC: "
        f"{cv_summary['metrics']['test_auroc']['mean']:.4f} ± "
        f"{cv_summary['metrics']['test_auroc']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Test AUPRC: "
        f"{cv_summary['metrics']['test_auprc']['mean']:.4f} ± "
        f"{cv_summary['metrics']['test_auprc']['std']:.4f}",
    )
    write_log(
        log_file,
        "  Test F1: "
        f"{cv_summary['metrics']['test_f1']['mean']:.4f} ± "
        f"{cv_summary['metrics']['test_f1']['std']:.4f}",
    )
    write_log(log_file, f"Saved training history to {history_path}")
    write_log(log_file, f"Saved fold summary to {fold_summary_path}")
    write_log(log_file, f"Saved CV summary to {cv_summary_path}")
    write_log(log_file, f"Training completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
