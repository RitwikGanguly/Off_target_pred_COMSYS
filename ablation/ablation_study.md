# Ablation Study: Reviewer-Focused Summary

## 1. Scope

This document reports the core ablation findings used for final model selection in the manuscript. It includes only the key results requested for reviewer-facing publication material.

## 2. Overall Findings

1. The negative-to-positive sampling ratio of **3:1** provides the strongest overall performance among tested class-balance settings.
2. In the architecture comparison, **h96-o96-L3-H4** is the top model across validation AUPRC, AUROC, and F1.

---

## 3. Negative-to-Positive Ablation Ratio

| Neg:Pos | Best val AUPRC | Calibrated threshold | Calibrated F1 | Composite score |
|---|---:|---:|---:|---:|
| 3:1 | 0.9962 | 0.1267 | 0.9784 | **0.9903** |
| 5:1 | 0.9955 | 0.0828 | 0.9746 | 0.9884 |
| 10:1 | 0.9921 | 0.2109 | 0.9673 | 0.9851 |
| 20:1 | 0.9810 | 0.1070 | 0.9630 | 0.9739 |

Selected setting for production: **3:1 (neg:pos)**.

---

## 4. Architecture-Metric Table

| Architecture | Val AUPRC | Val AUROC | Val F1 |
|---|---:|---:|---:|
| h96-o96-L3-H4 (main) | **0.9986** | **0.9996** | **0.9844** |
| h192-o64-L2-H2 | 0.9984 | 0.9994 | 0.9743 |
| h192-o64-L2-H4 | 0.9984 | 0.9994 | 0.9738 |
| h192-o192-L3-H2 | 0.9983 | 0.9994 | 0.9738 |
| h192-o192-L3-H8 | 0.9982 | 0.9993 | 0.9745 |
| h96-o96-L3-H2 | 0.9981 | 0.9994 | 0.9712 |
| h192-o192-L3-H4 | 0.9981 | 0.9993 | 0.9745 |
| h128-o96-L4-H2 | 0.9981 | 0.9993 | 0.9739 |
| h128-o96-L4-H4 | 0.9981 | 0.9993 | 0.9742 |
| h192-o128-L4-H4 | 0.9981 | 0.9993 | 0.9698 |

Interpretation: the selected manuscript model (**h96-o96-L3-H4**) is the best-performing architecture on all three validation metrics in Table 6.

---

## 5. Final Manuscript Model

### 5.1 Final production configuration

| Parameter | Value |
|---|---|
| Split | Inductive (guide-level) |
| Neg:Pos | 3:1 |
| MP/Supervision | 90% / 10% |
| Hidden / Out | 96 / 96 |
| Layers / Heads | 3 / 4 |
| Dropout / Edge dropout | 0.4 / 0.3 |
| LR / Weight decay | 1e-3 / 5e-3 |
| Loss | FocalLoss(alpha=0.25, gamma=2.0) |
| Operating threshold | 0.0779 |

### 5.2 Final model validation metrics

| Metric | Value |
|---|---:|
| Val AUPRC | **0.9986** |
| Val AUROC | **0.9996** |
| Val F1 | **0.9844** |
