import os
import warnings
warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM, Bidirectional, Dropout, Concatenate, Activation
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             classification_report, confusion_matrix, roc_curve,
                             f1_score, precision_score, recall_score)

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = "/home/bernadettem/TNBC/bgnmf_benchmarking/chemi/off_target1/benchmarking/crispr_net/crispr_off_T"
DATA_PATH = os.path.join(BASE_DIR, "crispr_off_T_combined.csv")

print("=" * 70)
print("CRISPR-Net Benchmarking on crispr_off_T Data")
print("Task: Predict ON-target (label=0) vs OFF-target (label=1)")
print("=" * 70)

# ============================================================
# 1. Sequence Encoder (7-dimensional, from CRISPR-Net paper)
# ============================================================
class SequenceEncoder:
    def __init__(self, on_seq, off_seq, with_category=False, label=None):
        tlen = 24
        if len(on_seq) > tlen:
            on_seq = on_seq[-tlen:]
        if len(off_seq) > tlen:
            off_seq = off_seq[-tlen:]
        self.on_seq = "-" * (tlen - len(on_seq)) + on_seq
        self.off_seq = "-" * (tlen - len(off_seq)) + off_seq
        
        self.encoded_dict_indel = {
            'A': [1, 0, 0, 0, 0],
            'T': [0, 1, 0, 0, 0],
            'G': [0, 0, 1, 0, 0],
            'C': [0, 0, 0, 1, 0],
            '_': [0, 0, 0, 0, 1],
            '-': [0, 0, 0, 0, 0],
            'N': [0, 0, 0, 0, 0]
        }
        
        self.direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1, '-': 0, 'N': 0}
        
        if with_category:
            self.label = label
        
        self.encode_on_off_dim7()
    
    def encode_sgRNA(self):
        code_list = []
        sgRNA_bases = list(self.on_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(self.encoded_dict_indel[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)
    
    def encode_off(self):
        code_list = []
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(self.encoded_dict_indel[off_bases[i]])
        self.off_code = np.array(code_list)
    
    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            
            on_b = on_bases[i]
            off_b = off_bases[i]
            
            if on_b == "N":
                on_b = off_b
            
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        
        self.on_off_code = np.array(on_off_dim7_codes)


def prepare_crispr_net_data(dataframe, guide_col='Guide_sequence',
                            target_col='Target_sequence', label_col='label'):
    encoded_sequences = []
    labels = []
    valid_indices = []
    
    for idx, row in dataframe.iterrows():
        try:
            guide_seq = str(row[guide_col]).upper()
            target_seq = str(row[target_col]).upper()
            
            encoder = SequenceEncoder(
                on_seq=guide_seq,
                off_seq=target_seq,
                with_category=True,
                label=row[label_col]
            )
            
            encoded_sequences.append(encoder.on_off_code)
            labels.append(row[label_col])
            valid_indices.append(idx)
            
        except Exception as e:
            if idx % 1000 == 0:
                print(f"  Error encoding row {idx}: {e}")
            continue
    
    X = np.array(encoded_sequences)
    X = X.reshape((len(X), 1, 24, 7))
    y = np.array(labels)
    return X, y, valid_indices


# ============================================================
# 2. CRISPR-Net Model Architecture (default params from paper)
#    Reference: Lin, J. & Wong, K.C. Off-target predictions in
#    CRISPR-Cas9 gene editing using deep learning. Bioinformatics
#    34, i656-i663 (2018).
#    GitHub: https://github.com/JasonLinjc/CRISPR-Net
# ============================================================
def conv2d_bn(x, filters, kernel_size, strides=1, padding='same',
              activation='relu', use_bias=True, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,
               use_bias=use_bias, name=name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def build_CRISPR_Net_model():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    
    branch_0 = conv2d_bn(inputs, 10, (1, 1), name='branch_0')
    branch_1 = conv2d_bn(inputs, 10, (1, 2), name='branch_1')
    branch_2 = conv2d_bn(inputs, 10, (1, 3), name='branch_2')
    branch_3 = conv2d_bn(inputs, 10, (1, 5), name='branch_3')
    
    branches = [inputs, branch_0, branch_1, branch_2, branch_3]
    mixed = Concatenate(axis=-1, name='concat')(branches)
    
    mixed = Reshape((24, 47), name='reshape_lstm')(mixed)
    
    blstm_out = Bidirectional(
        LSTM(15, return_sequences=True, name="LSTM_layer"),
        name='bidirectional_lstm'
    )(mixed)
    
    blstm_out = Flatten(name='flatten')(blstm_out)
    
    x = Dense(80, activation='relu', name='dense_80')(blstm_out)
    x = Dense(20, activation='relu', name='dense_20')(x)
    x = Dropout(rate=0.35, name='dropout')(x)
    
    prediction = Dense(1, activation='sigmoid', name='main_output')(x)
    
    model = Model(inputs=inputs, outputs=prediction, name='CRISPR_Net')
    return model


# ============================================================
# 3. Load and Prepare Data with PAIR-LEVEL SPLIT (no leakage)
# ============================================================
print("\n[1/5] Loading combined dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape: {df.shape}")
print(f"  Label distribution:\n{df['label'].value_counts().to_dict()}")
print(f"  Unique guides: {df['Guide_sequence'].nunique()}")

print("\n[2/5] Encoding sequences...")
X, y, valid_indices = prepare_crispr_net_data(df)
print(f"  Encoded data shape: {X.shape}")
print(f"  Labels shape: {y.shape}")

df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)

print("\n  Splitting by unique (guide, target) pairs to prevent data leakage...")
print("  (No identical pair appears in both train and test)")

df_valid['pair_key'] = df_valid['Guide_sequence'] + '|' + df_valid['Target_sequence']
unique_pairs = df_valid['pair_key'].unique()
pair_labels = df_valid.drop_duplicates(subset='pair_key').set_index('pair_key')['label']

train_pairs, test_pairs = train_test_split(
    unique_pairs, test_size=0.2, random_state=42,
    stratify=pair_labels
)

print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

train_mask = df_valid['pair_key'].isin(train_pairs)
test_mask = df_valid['pair_key'].isin(test_pairs)

X_train = X[train_mask.values]
y_train = y[train_mask.values]
X_test = X[test_mask.values]
y_test = y[test_mask.values]

print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")
print(f"  Train label dist: {np.bincount(y_train)}")
print(f"  Test label dist:  {np.bincount(y_test)}")

train_pairs_set = set(train_pairs)
test_pairs_set = set(test_pairs)
overlap = train_pairs_set & test_pairs_set
print(f"  Pair overlap between train/test: {len(overlap)} (should be 0)")


# ============================================================
# 4. Build and Train Model (default params from paper)
#    - Optimizer: Adam with lr=0.0001
#    - Loss: binary_crossentropy
#    - Batch size: 10000
#    - Epochs: 100
# ============================================================
print("\n[3/5] Building CRISPR-Net model...")
model = build_CRISPR_Net_model()
model.summary()

adam_optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(
    loss='binary_crossentropy',
    optimizer=adam_optimizer,
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

BATCH_SIZE = 10000
EPOCHS = 100

print(f"\n[4/5] Training model (batch_size={BATCH_SIZE}, epochs={EPOCHS})...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)
print("  Training completed!")


# ============================================================
# 5. Evaluation on Test Set
# ============================================================
print("\n[5/5] Evaluating on test set...")
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

auroc = roc_auc_score(y_test, y_pred_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall_vals, precision_vals)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("CRISPR-Net Test Set Performance")
print("=" * 70)
print(f"  AUROC:  {auroc:.4f}")
print(f"  AUPRC:  {auprc:.4f}")
print(f"  F1:     {f1:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {cm}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['On-target (0)', 'Off-target (1)']))


# ============================================================
# 6. Save Results and Figures
# ============================================================
print("\nSaving results and figures...")

results = {
    "auroc": float(auroc),
    "auprc": float(auprc),
    "f1_score": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_test, y_pred, target_names=['On-target (0)', 'Off-target (1)']),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "learning_rate": 0.0001,
    "split_method": "unique (guide, target) pair split (no pair in both train and test)",
    "unique_pairs_train": int(len(train_pairs)),
    "unique_pairs_test": int(len(test_pairs)),
    "pair_overlap": int(len(overlap)),
    "task": "Predict ON-target (label=0) vs OFF-target (label=1)",
    "n_on_targets": int(np.sum(y == 0)),
    "n_off_targets": int(np.sum(y == 1))
}

with open(os.path.join(BASE_DIR, "crispr_net_results.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: crispr_net_results.json")

model_json = model.to_json()
with open(os.path.join(BASE_DIR, "crispr_net_model.json"), 'w') as f:
    f.write(model_json)
model.save_weights(os.path.join(BASE_DIR, "crispr_net.weights.h5"))
print(f"  Saved: crispr_net_model.json, crispr_net.weights.h5")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(history.history['auc'], label='Training AUC')
axes[2].plot(history.history['val_auc'], label='Validation AUC')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].set_title('Model AUC')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_history.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: training_history.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f'CRISPR-Net (AUC = {auroc:.4f})', linewidth=2, color='blue')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(recall_vals, precision_vals, label=f'CRISPR-Net (AUC = {auprc:.4f})', linewidth=2, color='red')
axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', label='Random Classifier', linewidth=1)
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "roc_pr_curves.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: roc_pr_curves.png")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['On-target (0)', 'Off-target (1)'],
            yticklabels=['On-target (0)', 'Off-target (1)'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix\nF1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: confusion_matrix.png")

print("\n" + "=" * 70)
print("All done! Files saved to:")
print(f"  {BASE_DIR}/")
print("=" * 70)
