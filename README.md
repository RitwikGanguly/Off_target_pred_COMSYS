# Off_target_pred_COMSYS

Reproducible code and data package for heterogeneous graph-based CRISPR off-target prediction.

## Data availability

All primary input datasets used in this repository are included directly under:

- `data/allframe_update_addEpige.txt` (CrisprOffT training source)
- `data/circle_seq_processed.csv` (CIRCLE-seq validation source)
- `data/all_off_target.csv` (CRISPRDeepOff validation source)

No external download is required for these three files.

## Reproduction workflow

Run from the repository root (`Off_target_pred_COMSYS/`):

1) Build graph and filtered training CSV

```bash
python model/graph_building.py
```

This generates:

- `data/offtarget_filtered.csv`
- `model/hetero_graph_data_new1.pt`

2) Train/evaluate the main model

```bash
python model/offtarget_pred_model.py
```

Default outputs:

- checkpoints -> `model/checkpoints/`
- figures -> `figs/`

## Notes on reproducibility

- Paths are repository-relative by default (portable across machines).
- Most key settings can be overridden with environment variables (e.g., `OFFTARGET_SEED`, `OFFTARGET_NUM_EPOCHS`, `OFFTARGET_CHECKPOINT_DIR`).
- If running on CPU-only systems, the scripts automatically fall back to CPU.
- k-mer fallback embeddings are deterministic across runs and machines.

## Optional environment overrides

- `OFFTARGET_RAW_DATA_PATH` -> raw source table (default: `data/allframe_update_addEpige.txt`)
- `OFFTARGET_FILTERED_CSV_PATH` -> filtered OFF-target table (default: `data/offtarget_filtered.csv`)
- `OFFTARGET_GRAPH_DATA_PATH` -> graph artifact path (default: `model/hetero_graph_data_new1.pt`)
- `CUDA_DEVICE` -> compute device selection (for example `cuda`, `cuda:0`, `cpu`)
