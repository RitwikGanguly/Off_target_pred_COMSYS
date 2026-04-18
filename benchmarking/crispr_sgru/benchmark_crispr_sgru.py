import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Bidirectional, Concatenate, Conv2D, Dense, Dropout, Flatten, GRU, Reshape
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

BASE_DIR = "/home/bernadettem/TNBC/bgnmf_benchmarking/chemi/off_target1/benchmarking/sgru/crispr_off_T"
SOURCE_DATA_PATH = "/home/bernadettem/TNBC/bgnmf_benchmarking/chemi/off_target1/data/allframe_update_addEpige.txt"
OFF_SAMPLE_SIZE = 5000
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.2
HARD_NEGATIVE_MAX_MISMATCH = 1
OFF_EXACT_MATCH_RATIO = 0.12

# Default training params aligned to Crispr-SGRU main repository
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.1


class CrisprSGRUEncoder:
    def __init__(self):
        self.encoded_dict_indel = {
            "A": [1, 0, 0, 0, 0],
            "T": [0, 1, 0, 0, 0],
            "G": [0, 0, 1, 0, 0],
            "C": [0, 0, 0, 1, 0],
            "_": [0, 0, 0, 0, 1],
            "-": [0, 0, 0, 0, 0],
        }
        self.direction_dict = {"A": 5, "G": 4, "C": 3, "T": 2, "_": 1}
        self.tlen = 24

    def encode(self, guide_seq, target_seq):
        guide_seq = "-" * (self.tlen - len(guide_seq)) + guide_seq
        target_seq = "-" * (self.tlen - len(target_seq)) + target_seq

        guide_bases = list(guide_seq)
        target_bases = list(target_seq)

        guide_code = []
        target_code = []

        for i, base in enumerate(guide_bases):
            b = base if base != "N" else target_bases[i]
            guide_code.append(self.encoded_dict_indel.get(b, [0, 0, 0, 0, 0]))
        for base in target_bases:
            target_code.append(self.encoded_dict_indel.get(base, [0, 0, 0, 0, 0]))

        guide_code = np.array(guide_code)
        target_code = np.array(target_code)

        on_off_dim7_codes = []
        for i in range(len(guide_bases)):
            diff_code = np.bitwise_or(guide_code[i], target_code[i])

            g_b = guide_bases[i]
            t_b = target_bases[i]

            if g_b == "N":
                g_b = t_b

            dir_code = np.zeros(2)
            if g_b != "-" and t_b != "-" and g_b in self.direction_dict and t_b in self.direction_dict:
                if self.direction_dict[g_b] != self.direction_dict[t_b]:
                    if self.direction_dict[g_b] > self.direction_dict[t_b]:
                        dir_code[0] = 1
                    else:
                        dir_code[1] = 1

            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))

        return np.array(on_off_dim7_codes)


def _split_with_optional_stratify(units, labels, test_size, random_state):
    units = np.asarray(units)
    labels = np.asarray(labels)
    stratify_labels = None
    if len(np.unique(labels)) > 1 and len(units) >= 2:
        stratify_labels = labels
    return train_test_split(
        units,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )


def build_crispr_sgru():
    # Architecture matches BrokenStringx/Crispr-SGRU Train/MODEL.py (Crispr_SGRU)
    inputs = Input(shape=(1, 24, 7), name="main_input")
    conv_1 = Conv2D(10, (1, 1), padding="same", activation="relu")(inputs)
    conv_2 = Conv2D(10, (1, 2), padding="same", activation="relu")(inputs)
    conv_3 = Conv2D(10, (1, 3), padding="same", activation="relu")(inputs)
    conv_4 = Conv2D(10, (1, 5), padding="same", activation="relu")(inputs)

    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)

    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2 = Reshape((24, 7))(inputs)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])

    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])

    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=DROPOUT_RATE)(x)
    outputs = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Crispr_SGRU")
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def load_and_sample_data():
    df = pd.read_csv(SOURCE_DATA_PATH, sep="\t", low_memory=False)
    df.columns = df.columns.str.strip()

    required_cols = ["Guide_sequence", "Target_sequence", "Identity", "Mismach"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in source data: {missing_cols}")

    df = df[required_cols].dropna().copy()
    df["Guide_sequence"] = df["Guide_sequence"].astype(str).str.upper().str.strip()
    df["Target_sequence"] = df["Target_sequence"].astype(str).str.upper().str.strip()
    df["Identity"] = df["Identity"].astype(str).str.upper().str.strip()
    df["Mismach"] = pd.to_numeric(df["Mismach"], errors="coerce")

    df = df[df["Identity"].isin(["ON", "OFF"]) & df["Mismach"].notna()].copy()
    df_on = df[df["Identity"] == "ON"].copy().reset_index(drop=True)
    df_off = df[df["Identity"] == "OFF"].copy().reset_index(drop=True)

    df_off_hard = df_off[df_off["Mismach"] <= HARD_NEGATIVE_MAX_MISMATCH].copy().reset_index(drop=True)
    df_off_exact = df_off_hard[df_off_hard["Mismach"] == 0].copy().reset_index(drop=True)
    df_off_near = df_off_hard[df_off_hard["Mismach"] > 0].copy().reset_index(drop=True)

    if len(df_off_hard) == 0:
        df_off_sampled = df_off_hard.copy()
    elif len(df_off_exact) > 0 and len(df_off_near) > 0:
        n_exact = int(round(OFF_SAMPLE_SIZE * OFF_EXACT_MATCH_RATIO))
        n_exact = max(0, min(n_exact, OFF_SAMPLE_SIZE))
        n_near = OFF_SAMPLE_SIZE - n_exact

        exact_sample = df_off_exact.sample(
            n=n_exact,
            replace=(n_exact > len(df_off_exact)),
            random_state=RANDOM_STATE,
        )
        near_sample = df_off_near.sample(
            n=n_near,
            replace=(n_near > len(df_off_near)),
            random_state=RANDOM_STATE,
        )
        df_off_sampled = pd.concat([exact_sample, near_sample], ignore_index=True)
        df_off_sampled = df_off_sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        n_off = OFF_SAMPLE_SIZE
        df_off_sampled = df_off_hard.sample(
            n=n_off,
            replace=(n_off > len(df_off_hard)),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    df_on["label"] = 0
    df_off_sampled["label"] = 1

    combined = pd.concat([df_on, df_off_sampled], ignore_index=True)
    combined = combined[["Guide_sequence", "Target_sequence", "Identity", "Mismach", "label"]]

    return combined, df_on, df_off_sampled, len(df_off_hard), len(df_off)


def encode_dataset(df):
    encoder = CrisprSGRUEncoder()
    encoded_sequences = []
    labels = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding", ncols=90):
        try:
            guide = str(row["Guide_sequence"]).upper()
            target = str(row["Target_sequence"]).upper()
            encoded = encoder.encode(guide, target)
            if encoded.shape == (24, 7):
                encoded_sequences.append(encoded)
                labels.append(int(row["label"]))
                valid_indices.append(idx)
        except Exception:
            continue

    X = np.array(encoded_sequences, dtype=np.float32).reshape((-1, 1, 24, 7))
    y = np.array(labels, dtype=np.int32)
    df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)
    return X, y, df_valid


def guide_level_split(df_valid):
    df_valid = df_valid.copy()
    guide_stats = df_valid.groupby("Guide_sequence")["label"].agg(["count", "sum"]) 
    guide_has_both_labels = (
        (guide_stats["sum"] > 0) & (guide_stats["sum"] < guide_stats["count"])
    ).astype(int)

    all_guides = guide_stats.index.to_numpy()
    all_guide_flags = guide_has_both_labels.values

    train_guides, test_guides = _split_with_optional_stratify(
        all_guides,
        all_guide_flags,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_flags = guide_has_both_labels.loc[train_guides].values
    train_guides, val_guides = _split_with_optional_stratify(
        train_guides,
        train_flags,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
    )

    return df_valid, train_guides, val_guides, test_guides


def save_training_plots(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history.get("loss", []), label="Training Loss")
    axes[0].plot(history.history.get("val_loss", []), label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Model Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history.get("accuracy", []), label="Training Accuracy")
    axes[1].plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Model Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_eval_plots(y_test, y_prob, auroc, precision_vals, recall_vals, auprc, cm, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"Crispr-SGRU (AUC = {auroc:.4f})", linewidth=2, color="blue")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall_vals, precision_vals, label=f"Crispr-SGRU (AUC = {auprc:.4f})", linewidth=2, color="red")
    axes[1].axhline(y=float(np.mean(y_test)), color="k", linestyle="--", label="Random Classifier", linewidth=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["On-target (0)", "Off-target (1)"],
        yticklabels=["On-target (0)", "Off-target (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("Crispr-SGRU Benchmarking on crispr_off_T Data")
    print("Task: Predict ON-target (label=0) vs OFF-target (label=1)")
    print("=" * 70)

    os.makedirs(BASE_DIR, exist_ok=True)

    print("\n[1/6] Loading and sampling dataset...")
    combined_df, on_df, off_df, off_hard_count, off_total_count = load_and_sample_data()
    print(f"  ON rows used:  {len(on_df)}")
    print(f"  OFF rows used: {len(off_df)}")
    print(
        "  OFF hard-negative pool "
        f"(Mismach <= {HARD_NEGATIVE_MAX_MISMATCH}): {off_hard_count}/{off_total_count}"
    )
    print(f"  Combined rows: {len(combined_df)}")

    combined_df.to_csv(os.path.join(BASE_DIR, "crispr_off_T_combined.csv"), index=False)
    on_df.to_csv(os.path.join(BASE_DIR, "generated_on_targets.csv"), index=False)
    off_df.to_csv(os.path.join(BASE_DIR, "sampled_off_targets.csv"), index=False)

    print("\n[2/6] Encoding sequences...")
    X, y, df_valid = encode_dataset(combined_df)
    print(f"  Encoded X shape: {X.shape}")
    print(f"  Labels shape:    {y.shape}")
    print(f"  Label counts:    ON={(y == 0).sum()}, OFF={(y == 1).sum()}")

    print("\n[3/6] Guide-level split (no leakage)...")
    df_valid, train_guides, val_guides, test_guides = guide_level_split(df_valid)

    train_mask = df_valid["Guide_sequence"].isin(set(train_guides)).values
    val_mask = df_valid["Guide_sequence"].isin(set(val_guides)).values
    test_mask = df_valid["Guide_sequence"].isin(set(test_guides)).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    guide_overlap_train_test = len(set(train_guides) & set(test_guides))
    guide_overlap_train_val = len(set(train_guides) & set(val_guides))
    guide_overlap_val_test = len(set(val_guides) & set(test_guides))

    print(f"  Train samples: {len(X_train)} | guides: {len(train_guides)}")
    print(f"  Val samples:   {len(X_val)} | guides: {len(val_guides)}")
    print(f"  Test samples:  {len(X_test)} | guides: {len(test_guides)}")
    print(
        "  Guide overlap (train/val/test): "
        f"{guide_overlap_train_val}/{guide_overlap_train_test}/{guide_overlap_val_test}"
    )

    y_train_cat = np.zeros((len(y_train), 2), dtype=np.float32)
    y_train_cat[np.arange(len(y_train)), y_train] = 1.0
    y_val_cat = np.zeros((len(y_val), 2), dtype=np.float32)
    y_val_cat[np.arange(len(y_val)), y_val] = 1.0
    print("\n[4/6] Building Crispr-SGRU model...")
    model = build_crispr_sgru()
    model.summary()

    print(f"\n[5/6] Training model (batch_size={BATCH_SIZE}, epochs={EPOCHS})...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    print("\n[6/6] Evaluating on test set...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_prob = y_pred_prob[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall_vals, precision_vals)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("Crispr-SGRU Test Set Performance")
    print("=" * 70)
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")

    results = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["On-target (0)", "Off-target (1)"]
        ),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "split_method": "Guide_sequence-level split into train/val/test (unseen guides in test)",
        "unique_guides_train": int(len(train_guides)),
        "unique_guides_val": int(len(val_guides)),
        "unique_guides_test": int(len(test_guides)),
        "guide_overlap_train_val": int(guide_overlap_train_val),
        "guide_overlap_train_test": int(guide_overlap_train_test),
        "guide_overlap_val_test": int(guide_overlap_val_test),
        "task": "Predict ON-target (label=0) vs OFF-target (label=1)",
        "n_on_targets": int((y == 0).sum()),
        "n_off_targets": int((y == 1).sum()),
        "hard_negative_max_mismatch": int(HARD_NEGATIVE_MAX_MISMATCH),
        "off_exact_match_ratio": float(OFF_EXACT_MATCH_RATIO),
        "hard_negative_pool_size": int(off_hard_count),
        "off_total_size": int(off_total_count),
        "source_data": SOURCE_DATA_PATH,
    }

    with open(os.path.join(BASE_DIR, "crispr_sgru_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_prob_off_target": y_prob,
            "y_pred": y_pred,
        }
    )
    predictions_df.to_csv(os.path.join(BASE_DIR, "test_predictions.csv"), index=False)

    pd.DataFrame([results]).drop(columns=["classification_report", "confusion_matrix"]).to_csv(
        os.path.join(BASE_DIR, "benchmark_results.csv"), index=False
    )

    model.save(os.path.join(BASE_DIR, "crispr_sgru_benchmark.keras"))

    save_training_plots(history, BASE_DIR)
    save_eval_plots(y_test, y_prob, auroc, precision_vals, recall_vals, auprc, cm, BASE_DIR)

    print("\nSaved outputs:")
    print(f"  {os.path.join(BASE_DIR, 'crispr_sgru_results.json')}")
    print(f"  {os.path.join(BASE_DIR, 'benchmark_results.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'test_predictions.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'crispr_sgru_benchmark.keras')}")
    print(f"  {os.path.join(BASE_DIR, 'training_history.png')}")
    print(f"  {os.path.join(BASE_DIR, 'roc_pr_curves.png')}")
    print(f"  {os.path.join(BASE_DIR, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
