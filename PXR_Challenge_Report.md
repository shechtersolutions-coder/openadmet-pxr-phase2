# OpenADMET PXR Blind Challenge — Modeling Report

## Overview

This report summarizes the end-to-end machine learning pipeline developed to predict
PXR (Pregnane X Receptor) pEC50 values for the OpenADMET Blind Challenge Phase 2.
The challenge required predicting pEC50 for 260 blinded test compounds using a training
set of dose-response data from Octant Bio.

---

## 1. Data Sources

| File | Description | Rows |
|---|---|---|
| `pxr-challenge_TRAIN.csv` | Primary dose-response training set | 4,139 |
| `pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv` | Semi-pure assay data | 96 |
| `pxr-challenge_single_concentration_TRAIN.csv` | Single-concentration screen | 21,003 |
| `pxr-challenge_counter-assay_TRAIN.csv` | PXR-null counter assay | 2,859 |
| `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv` | Unblinded test set (evaluation) | 253 |
| `pxr-challenge_TEST_BLINDED.csv` | Blinded test set (submission) | 513 |

---

## 2. Data Cleaning

### 2.1 Training Set Filters

Three filters were applied to the raw TRAIN data:

**Counter-assay filter** — Compounds that are potent in the primary assay (pEC50 ≥ 6)
but not selective (primary pEC50 − counter pEC50 < 1.5) were removed as likely
non-specific reporter activators. This removed 6 compounds.

**Emax outlier filter** — Compounds with Emax vs. positive control > 5 were removed
as assay artifacts. This removed 6 compounds.

**Semi-pure augmentation** — 91 additional compounds from the semi-pure assay file
were added where OCNT_ID was not already present in the TRAIN set. The corrected
semi-pure pEC50 was used as the target value.

**Final training set: 4,127 compounds**

### 2.2 pEC50 Distribution

| Bin | Count | % |
|---|---|---|
| < 4 | ~1,065 | 25.8% |
| 4–5 | ~1,476 | 35.8% |
| 5–6 | ~1,230 | 29.8% |
| > 6 | ~356 | 8.6% |

---

## 3. Feature Engineering

### 3.1 Chemistry Features (8,321 total)

| Feature type | Bits/Count | Description |
|---|---|---|
| Morgan r=2 fingerprint | 2,048 | Circular substructure, radius 2 |
| Morgan r=3 fingerprint | 2,048 | Circular substructure, radius 3 |
| MACCS keys | 167 | Known pharmacophoric patterns |
| RDKit path fingerprint | 2,048 | Linear path substructures, max path 6 |
| 2D descriptors | 8 | MolWt, LogP, TPSA, HBD, HBA, RotBonds, Rings, CSP3 |
| 3D descriptors | 10 | PMI1-3, NPR1-2, RadGyration, Asphericity, Eccentricity, ISF, Spherocity |

3D conformers were generated using ETKDGv3 (10 conformers) with MMFF optimization.
All features were cached to disk per SMILES to avoid recomputation.

### 3.2 Feature Leakage Investigation

Initial models included uncertainty features derived from the dose-response fit:
- `pEC50_ci_asymmetry` = (CI_upper + CI_lower) / 2 ≈ pEC50 itself → **removed (data leakage)**
- `pEC50_ci_width` → **removed (correlated with pEC50)**
- `pEC50_inv_stderr` → **removed (not available for test compounds)**

SP (single-concentration) features were initially top predictors but caused a massive
train→test gap because test compounds had 0% SP coverage vs 58.9% in training.
**SP features were dropped for the final submission model.**

---

## 4. Model Development

### 4.1 Validation Strategy

Scaffold-based split (Bemis-Murcko) was used throughout:
- 80% of scaffolds → training
- 20% of scaffolds → validation

This is stricter than random split and more representative of true generalization,
since 87.8% of test set scaffolds are unseen in training.

### 4.2 LightGBM Model

Hyperparameters were tuned using Optuna (100 trials, TPE sampler):

| Parameter | Value |
|---|---|
| learning_rate | 0.02149 |
| num_leaves | 146 |
| min_data_in_leaf | 53 |
| feature_fraction | 0.6561 |
| bagging_fraction | 0.9774 |
| bagging_freq | 7 |
| reg_alpha | 0.000517 |
| reg_lambda | 2.70e-06 |
| max_depth | 10 |
| num_boost_round | 2,000 |

Final model trained on ALL 4,127 training compounds (no held-out validation).

### 4.3 Chemprop Model (Message Passing Neural Network)

Trained using Chemprop v2.2.3 CLI with scaffold-balanced split:

| Parameter | Value |
|---|---|
| Architecture | BondMessagePassing MPNN |
| Hidden dim | 300 |
| Depth | 3 |
| FFN layers | 2 |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Split | scaffold_balanced |

---

## 5. Results

### 5.1 Performance on Unblinded Test Set (n=253)

| Model | MAE | RMSE | R² | Spearman ρ |
|---|---|---|---|---|
| LightGBM (chemistry only) | 0.510 | 0.746 | 0.478 | 0.762 |
| Chemprop MPNN | 0.572 | 0.766 | 0.448 | 0.738 |
| **Ensemble (LGBM + Chemprop)** | **0.500** | **—** | **0.492** | **0.778** |

### 5.2 Performance by pEC50 Bin

| Bin | n | MAE | Bias | Notes |
|---|---|---|---|---|
| < 4 | 55 | 1.091 | −0.982 | Model overpredicts inactives |
| 4–5 | 86 | 0.358 | −0.068 | Excellent |
| 5–6 | 102 | 0.307 | +0.232 | Excellent |
| 6–7 | 10 | 0.709 | +0.709 | Underpredicts potent compounds |

### 5.3 Key Findings

- The **pEC50 < 4 overprediction** is the largest source of error. These compounds are
  structurally diverse inactives that the model conflates with weak actives.
- **87.8% of test scaffolds are unseen** in training — the dominant driver of the
  validation→test gap (scaffold val R²=0.829 vs test R²=0.478).
- **SP features gave artificially high validation R²** (0.829) but collapsed on test
  (R²=0.303) due to 0% coverage — a critical lesson in feature leakage via coverage asymmetry.
- **Chemprop MPNN generalizes better** than LightGBM on test (Spearman 0.738 vs 0.587)
  because it learns task-specific atom representations rather than relying on fixed fingerprints.
- **Ensembling** the two models improved all metrics, confirming complementary strengths.

---

## 6. Submission

**File:** `pxr_submission_best_ensemble.csv`  
**Format:** SMILES, Molecule Name, pEC50 (513 rows)  
**Method:** Weighted average — LightGBM × best_weight + Chemprop × (1 − best_weight)

---

## 7. What Would Improve the Model

Ranked by expected impact:

| Approach | Expected gain | Complexity |
|---|---|---|
| UniMol pretrained embeddings | +0.05–0.10 R² | High |
| ChEMBL PXR data augmentation (~800 compounds) | +0.03–0.08 R² | Medium |
| Multi-seed Chemprop ensemble (5 models) | +0.02–0.05 R² | Low |
| Mordred full descriptor set (1,800+) | +0.01–0.03 R² | Low |
| GNINA/docking features (PXR crystal structures) | +0.03–0.07 R² | Very high |

Top leaderboard entries (MAE ~0.43, Spearman ~0.83) used Chemprop + UniMol + TabICL
ensembles with docking-based features from PXR crystal structures.

---

## 8. Repository Structure

```
PXR-Challenge-Tutorial/
├── inputs/                          # Raw data files
├── feature_cache_final/             # Cached molecular features (npz)
├── chemprop_output/                 # Chemprop training output + best.pt
├── chemprop_train.csv               # Training CSV for Chemprop CLI
├── chemprop_preds_unb.csv           # Chemprop unblinded predictions
├── chemprop_preds_bl.csv            # Chemprop blinded predictions
├── unblinded_test_predictions_FINAL.csv  # LightGBM unblinded predictions
├── pxr_lgbm_final_blinded.csv       # LightGBM blinded predictions
└── pxr_submission_best_ensemble.csv # Final submission file
```
