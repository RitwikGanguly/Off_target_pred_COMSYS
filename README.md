## An Edge-Aware Heterogeneous Graph Transformer for Biologically Informed Prediction of CRISPR–Cas9 Off-Target Sites


![Model workflow for edge-aware heterogeneous graph transformer](model_workflow.png)

## Overview

This repository implements a biologically informed off-target prediction framework that models guide RNAs and candidate genomic sites as a heterogeneous graph. The main model performs edge-aware message passing and edge-level scoring to estimate off-target risk, with explicit support for inductive guide-level evaluation and calibrated decision thresholds.

## Key methodological elements

- Heterogeneous bipartite graph formulation (`guide` and `site` node types).
- Edge-aware TransformerConv message passing with reverse-edge propagation.
- Rich 25-dimensional edge descriptors (position-wise matching and aggregate biophysical features).
- Inductive guide-level split and disjoint message-passing/supervision edges.
- Focal-loss training for class imbalance and threshold-calibrated inference.

## Repository structure

- `model/` - graph construction and main model training/evaluation.
- `data/` - core training and external validation datasets.
- `validation/` - external validation scripts (CIRCLE-seq and CRISPRDeepOff).
- `benchmarking/` - baseline benchmarking assets (CRISPR-Net, CRISPR-SGRU, CCLMoff).
- `ablation/` - manuscript-focused ablation summary.

## Data availability

Primary datasets required for end-to-end reproduction are included in-repo:

- `data/allframe_update_addEpige.txt` (training source)
- `data/circle_seq_processed.csv` (CIRCLE-seq validation source)
- `data/all_off_target.csv` (DeepCrispr validation source)

No external download is required for these three files.

## Environment setup

Create a Python environment and install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn tqdm biopython
pip install torch torch-geometric transformers
```

Optional dependencies for extended experiments:

- `multimolecule` (RNA-FM tokenizer/model path)
- `tensorflow` / `keras` (CRISPR-Net and CRISPR-SGRU baselines)
- `rna-fm` (CCLMoff baseline)

## Reproducing the main pipeline

Run from the repository root (`Off_target_pred_COMSYS/`):

1. Build filtered training data and heterogeneous graph artifacts

```bash
python model/graph_building.py
```

Expected outputs:

- `data/offtarget_filtered.csv`
- `model/hetero_graph_data_new1.pt`

2. Train and evaluate the primary model

```bash
python model/offtarget_pred_model.py
```

Default outputs:

- checkpoints in `model/checkpoints/`
- figures in `figs/`

## Final manuscript configuration

The final production setting used for manuscript reporting is:

| Parameter | Value |
|---|---|
| Split strategy | Inductive (guide-level) |
| Negative:positive ratio | 3:1 |
| Message-passing:supervision split | 90% : 10% |
| Hidden / output channels | 96 / 96 |
| Layers / heads | 3 / 4 |
| Dropout / edge dropout | 0.4 / 0.3 |
| Optimizer settings | LR 1e-3, weight decay 5e-3 |
| Loss | FocalLoss (alpha=0.25, gamma=2.0) |
| Operating threshold | 0.0779 |

Validation metrics for this configuration:

| Metric | Value |
|---|---:|
| Validation AUPRC | 0.9986 |
| Validation AUROC | 0.9996 |
| Validation F1 | 0.9844 |
| Test F1 | 0.9669 |
| Precision | 0.9859 |
| Recall | 0.9487 |
| Accuracy | 0.9838 |

For the compact reviewer-facing ablation summary, see `ablation/ablation_study.md`.

## External validation and baseline benchmarking

Representative scripts are provided under:

- `validation/model_validation_circle_seq.py`
- `validation/model_validation_crisprdeepoff.py`
- `benchmarking/`

Some validation/benchmark scripts contain experiment-local path settings and may require path updates before execution in a new environment.

## Reproducibility controls

Core scripts support environment-variable overrides for data paths, model hyperparameters, and runtime behavior. Common examples include:

- `OFFTARGET_RAW_DATA_PATH`
- `OFFTARGET_FILTERED_CSV_PATH`
- `OFFTARGET_GRAPH_DATA_PATH`
- `OFFTARGET_SEED`
- `OFFTARGET_NUM_EPOCHS`
- `OFFTARGET_CHECKPOINT_DIR`
- `OFFTARGET_FIG_DIR`
- `CUDA_DEVICE`

## License

This project is released under the MIT License. See `LICENSE`.
