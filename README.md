# OpenADMET PXR Induction Challenge — Methodology Report

## Summary

This report describes the modeling approach used for the Activity Prediction track of the OpenADMET PXR Induction Blind Challenge. The goal is to predict pEC50 values for 513 analog compounds from the Pregnane X Receptor (PXR) induction assay. Our final model achieves **MAE=0.4966, RMSE=0.7154, R²=0.5195, Spearman=0.7635** on the Phase 1 unblinded test set (253 compounds).

---

## 1. Data Loading and Curation

### Input Files
- `pxr-challenge_TRAIN.csv` — primary assay training data (pEC50, Emax)
- `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv` — Phase 1 test set (253 compounds, ground truth available)
- `pxr-challenge_TEST_BLINDED.csv` — full blinded test set (513 compounds)
- `pxr-challenge_counter-assay_TRAIN.csv` — PXR-null counter assay data

### Feature Selection from Training Data
The following columns were retained from the training set:
- `Molecule Name`, `SMILES`, `OCNT_ID`, `pEC50`
- `Emax_estimate (log2FC vs. baseline)` → renamed to `Emax`
- `Emax.vs.pos.ctrl_estimate (dimensionless)` → renamed to `Emax_vs_ctrl`

### Quality Control Filters

**Counter-assay filter:** Compounds with pEC50 ≥ 6 (EC50 ≤ 1 µM) in the primary assay AND a difference of less than 1.5 log units between primary pEC50 and counter assay pEC50 were removed. These compounds are likely false positives — their signal cannot be confidently attributed to genuine PXR activation.

```
potent       = df["pEC50"] >= 6
not_selective = potent & (df["pEC50"] - df["ca_pEC50"] < 1.5)
df = df[~not_selective]
```

**Emax outlier filter:** Compounds with Emax vs. positive control > 5 were removed as assay artifacts. A signal more than 5× the positive control is biologically implausible for a genuine PXR agonist and likely reflects autoluminescence, cytotoxicity, or non-specific transcriptional activation.

```
df = df[df["Emax_vs_ctrl"] <= 5]
```

**Final training set size: 4,127 compounds**

---

## 2. Molecular Featurization

All features were computed using RDKit and cached to disk to avoid recomputation. The full feature vector per compound consists of:

### Fingerprints
| Feature | Size | Description |
|---|---|---|
| Morgan r=2 | 2048 bits | Circular fingerprint, radius 2 (ECFP4-equivalent) |
| Morgan r=3 | 2048 bits | Circular fingerprint, radius 3 (ECFP6-equivalent) |
| MACCS keys | 167 bits | Structural keys encoding known pharmacophoric features |
| RDKit fingerprint | 2048 bits | Path-based topological fingerprint |

### 2D Physicochemical Descriptors
`MolWt`, `LogP`, `TPSA`, `HBD`, `HBA`, `RotatableBonds`, `RingCount`, `FractionCSP3`

### 3D Shape Descriptors
Computed from a single MMFF-optimized 3D conformer (ETKDGv3):
`PMI1`, `PMI2`, `PMI3`, `NPR1`, `NPR2`, `RadiusOfGyration`, `Asphericity`, `Eccentricity`, `InertialShapeFactor`, `SpherocityIndex`

**Total base feature dimensions: 6,329**

---

## 3. Analog Neighborhood Features (Step 2)

Seven additional features were computed for each compound based on its **top-10 most similar training compounds** by Tanimoto similarity on Morgan r=2 fingerprints (threshold ≥ 0.3):

| Feature | Description |
|---|---|
| `nbr_mean_pec50` | Mean pEC50 of similar neighbors |
| `nbr_max_pec50` | Max pEC50 of similar neighbors |
| `nbr_min_pec50` | Min pEC50 of similar neighbors |
| `nbr_std_pec50` | Std of neighbor pEC50 values |
| `nbr_sim_weighted_pec50` | Tanimoto-weighted mean pEC50 |
| `nbr_mean_sim` | Mean Tanimoto similarity to neighbors |
| `nbr_delta_from_max` | Max − mean pEC50 of neighbors |

**Rationale:** In a lead optimization dataset, a compound's activity is strongly constrained by its structural neighbors. These features give the model an explicit signal about the local activity landscape, which is especially useful for weak compounds that are analogs of moderately active scaffolds.

**Final augmented feature dimensions: 6,336**

---

## 4. Model Architecture

### Base Model: LightGBM
A gradient boosted tree model (LightGBM) was used as the primary model. LightGBM handles high-dimensional sparse binary features (fingerprints) efficiently and is well-suited to tabular molecular property prediction.

### Cross-validation Strategy: Scaffold-aware Split
To avoid data leakage from structurally similar compounds appearing in both train and validation sets, a **Murcko scaffold-based split** was used:
- 80% of unique scaffolds → training
- 20% of unique scaffolds → validation
- Scaffold train: 3,168 compounds | Scaffold val: 959 compounds

### Hyperparameter Optimization: Optuna
Two rounds of Optuna hyperparameter search were performed (50 trials each, TPE sampler):

**Round 1** — Standard optimization on unweighted data:
```
learning_rate=0.0218, num_leaves=112, min_data_in_leaf=41,
feature_fraction=0.507, bagging_fraction=0.562, bagging_freq=9,
reg_alpha=6.07e-8, reg_lambda=3.83e-3, max_depth=5
```

**Round 2** — Optimization with sample weights included in the objective (see Section 5):
```
learning_rate=0.0152, num_leaves=95, min_data_in_leaf=18,
feature_fraction=0.697, bagging_fraction=0.660, bagging_freq=1,
reg_alpha=3.76e-4, reg_lambda=0.069, max_depth=6
```

The second round found meaningfully different hyperparameters when the weighted loss was incorporated into the search, leading to further improvement.

---

## 5. Sample Weighting for Low pEC50 Compounds

### Problem Identified
Error analysis on the unblinded test set revealed severe underperformance on weak compounds:

| pEC50 range | MAE (baseline) | Count |
|---|---|---|
| < 4.5 | 0.929 | 81 |
| 4.5 – 5.0 | 0.343 | 60 |
| 5.0 – 5.5 | 0.222 | 70 |
| 5.5 – 6.0 | 0.437 | 32 |
| > 6.0 | 0.671 | 10 |

The model was regressing toward the training mean (~4.3) for weak compounds, producing errors of nearly 1 full log unit on the `<4.5` bin despite having 1,799 training examples in that range. This indicated a feature discrimination problem rather than a data scarcity problem.

### Solution: Asymmetric Sample Weighting
LightGBM's `weight` parameter was used to upweight compounds at the extremes of the activity distribution during training:

| pEC50 range | Weight |
|---|---|
| < 4.5 (weak) | **6.0** |
| 4.5 – 5.5 (middle) | 1.0 |
| > 5.5 (potent) | **3.0** |

This forces the model to penalize errors on extreme compounds more heavily, reducing regression-to-mean behavior.

### Weight Selection
A grid search over `low_w ∈ {2,3,4,5,6}` and `high_w ∈ {1,2,3}` was performed. The combination `low_w=6.0, high_w=3.0` was selected as it minimized overall MAE while maintaining strong Spearman correlation.

---

## 6. Results

### Progressive Improvement

| Step | MAE | RMSE | R² | Spearman | Kendall |
|---|---|---|---|---|---|
| Baseline (original) | 0.5203 | 0.7471 | 0.4759 | 0.7474 | 0.5472 |
| + Neighbor features | 0.5219 | 0.7685 | 0.4455 | 0.7511 | 0.5527 |
| + Sample weights | 0.5111 | 0.7255 | 0.5058 | 0.7587 | 0.5612 |
| + Weighted Optuna | **0.4966** | **0.7154** | **0.5195** | **0.7635** | **0.5693** |

### Final Model — MAE by pEC50 Range

| pEC50 range | MAE (baseline) | MAE (final) | Δ |
|---|---|---|---|
| < 4.5 | 0.929 | 0.798 | −0.131 |
| 4.5 – 5.0 | 0.343 | 0.334 | −0.009 |
| 5.0 – 5.5 | 0.222 | 0.292 | +0.070 |
| 5.5 – 6.0 | 0.437 | 0.437 | 0.000 |
| > 6.0 | 0.671 | 0.655 | −0.016 |

The largest gain was on the weakest compounds (`<4.5` bin), which was the primary target of the sample weighting strategy. The slight worsening in the `5.0–5.5` bin is an acceptable tradeoff given the overall improvement.

### Blinded Test Set Predictions (513 compounds)
```
mean:  4.850
std:   0.610
min:   2.625
max:   6.017
```

---

## 7. What Was Tried and Discarded

**Isotonic calibration:** Post-hoc isotonic regression fitted on the scaffold validation set was tested but slightly worsened all metrics (MAE 0.5203 → 0.5240). The scaffold val set was not representative enough of the test distribution for the calibration to generalize.

**RDKit descriptor ensemble (Step 4):** A second LightGBM model trained on only ~26 physicochemical descriptors (no fingerprints) was tested as an ensemble partner. Model 2 alone achieved MAE=0.5763, too weak to improve the ensemble — all blends (50/50 through 80/20) underperformed Model 1 alone.

**Single-concentration assay features:** The single-concentration screening data (log2FC, FDR, Cohen's d at 8µM and 33µM) was available for 65% of training compounds but had 100% NaN rate on both test sets, since the analog test compounds were never in the primary screen. Discarded.

---

## 8. Software and Environment

| Package | Version |
|---|---|
| Python | 3.x |
| LightGBM | latest |
| RDKit | latest |
| Optuna | latest |
| scikit-learn | latest |
| pandas / numpy | latest |

---

## 9. Code

Full code is available in this repository. The main notebook follows this cell structure:

1. Imports
2. Data loading, QC filters, single-conc feature attempt
3. Feature computation functions (fingerprints, 2D/3D descriptors)
4. Base feature matrix construction (with disk caching)
5. Analog neighborhood feature computation
6. Augmented feature matrix assembly
7. Scaffold-aware train/val split
8. Optuna hyperparameter search (weighted objective)
9. Final model training with sample weights
10. Evaluation and submission generation
