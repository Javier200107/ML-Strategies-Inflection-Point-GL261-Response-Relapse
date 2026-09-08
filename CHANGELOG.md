# Changelog

All notable changes to this project are documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Done (paper revision — notebook `09_final_pipeline_PAPER.ipynb`)
- **A1 DONE** — Feature selection (variance + Spearman + Mann-Whitney + original_ heuristic) moved inside GLOO folds, fitted on training subjects only (no data leakage)
- **A2 DONE** — Animal-level majority-vote metrics added (AUC=0.92, Sens=0.60, Spec=1.00, PPV=1.00, Acc=0.80 across 10 mice)
- **A3 DONE** — Stratified temporal analysis early/mid/late terciles by day_of_study (AUC: 0.576 / 0.803 / 0.962)
- **A4 DONE** — Bootstrap 95% CIs (10,000 resamples) on exam-level AUC, Sensitivity, Specificity, PPV, Accuracy

### Pending (paper revision)
- Correct "outperforms across all metrics" claim in Discussion
- Add EfficientNet longitudinal limitation to Methods/Discussion
- Discuss clinical implications of low PPV
- Verify supplementary Tables S3/S4 appear in compiled PDF

---

## [0.2.0] — 2026-09-09

### Added
- `pyproject.toml` — project declared with uv; all dependencies listed (xgboost, sklearn, imbalanced-learn, pyradiomics, TF 2.10, opencv, etc.)
- `.python-version` — pinned to Python 3.10 (required by TensorFlow 2.10)
- `PAPER_CODE_MAPPING.md` — full traceability map from paper Table 1 results to code files and output directories
- `docs/paper/review/REVIEWER_RESPONSE.md` — point-by-point response to all reviewer comments (major + minor)
- `docs/paper/review/ACTION_PLAN.md` — 3-week action plan with exact code snippets for each change needed for resubmission
- `outputs/` — reorganised results directory with human-readable names (see Changed)
- `config.py` — added `OUTPUTS_RADIOMICS`, `OUTPUT_EXP_09`, `OUTPUTS_DL`, `OUTPUTS_GRADCAM`, `OUTPUTS_FIGURES`, `TFRECORD_NO_CONTROL`

### Changed
- **Repository structure fully reorganised:**
  - `src/begin/ordered/hyp3 copy/0-8_*.ipynb` → `notebooks/radiomics/01_–09_final_pipeline_PAPER.ipynb`
  - `src/begin/ordered/hyp3 copy/dl_extractfeatures_img_xgboost.ipynb` → `notebooks/deep_learning/01_efficientnet_feature_extraction_PAPER.ipynb`
  - `src/begin/ordered/hyp3 copy/hypotesis1_newds copy_good2 supergood copy 10.ipynb` → `notebooks/deep_learning/02_efficientnet_fine_tuning_PAPER.ipynb`
  - `src/notebooks/dl/gradcam.py` → `notebooks/deep_learning/gradcam.py`
  - `tfrecordhandler*.py` (root) → `src/`
  - `output_final1/hyp3/baseline_no_control_heuristic_.../` → `outputs/radiomics/gloo/09_final_pipeline_PAPER/`
  - `output_final1/DL/` → `outputs/deep_learning/gradcam/`
  - `output_final1/imgs/`, `output_final1/hyp3/imgs/` → `outputs/figures/`
  - `output_final1/hyp1/` → `outputs/radiomics/expanding_window/`
- `README.md` — rewritten with full project structure, paper results table, and uv setup instructions

### Archived (moved to `archive/`)
- ~70 exploratory/duplicate notebooks from `src/begin/`, `src/notebooks/`, `src/notebooks/dl/`
- Intermediate CSVs from repository root and `csvs/` folder
- Alternative DL channel ablation notebooks (`dl_fe_imgimgmask`, `_imgmaskmask`, etc.)
- VGG16 experiments (`hypotesis1OK.ipynb` and variants)
- NGBOOST and Random Forest alternative model notebooks
- Undersample variant notebook

---

## [0.1.3] — 2026-09-02

### Added
- Final experiment results committed (`output_final1/`)
- Additional intermediate datasets

### Changed
- Dataset updates for DL pipeline

---

## [0.1.2] — 2025-04-10

### Added
- Additional CSV datasets and feature exports

---

## [0.1.1] — 2025-03-27

### Changed
- Iterative improvements to radiomics GLOO pipeline (experiment notebooks 5–8)
- Feature selection refinements (heuristic filter for wavelet/LoG features)

---

## [0.1.0] — 2025-01-15

### Added
- Deep learning pipeline: EfficientNetB0 feature extraction + XGBoost (GLOO)
- EfficientNetB0 fine-tuning per fold (GLOO)
- `TFRecordDataHandlerDL` for multimodal image+mask TFRecords
- Grad-CAM analysis (`gradcam.py`) applied to mouse C1263 across days 11–19

### Changed
- Consolidated experiments into `src/begin/ordered/hyp3 copy/` (numbered 0–11)

---

## [0.0.9] — 2024-12-26

### Added
- Misclassification analysis by day of study and by mouse group
- Feature importance aggregation across GLOO folds
- `src/balance.py` — SMOTE, undersampling, and ensemble balancing helpers

---

## [0.0.8] — 2024-12-16

### Added
- Reorganisation of notebooks by hypothesis (`src/begin/ordered/`)
- `src/datasplitting.py` — `grouped_combinations_split_with_sampling()` (GLOO/leave-one-mouse-out)

---

## [0.0.7] — 2024-12-10

### Added
- GLOO cross-validation with per-fold misclassified/well-classified CSV export
- Per-fold feature importance CSV export

### Changed
- `src/processing.py` — `Processing` class refactored; added `drop_late_days()`, `drop_mice_group()`

---

## [0.0.6] — 2024-11-28

### Added
- Nested cross-validation for hyperparameter tuning (GridSearchCV + TimeSeriesSplit)
- XGBoost hyperparameter grid (Table S2 in paper)

---

## [0.0.5] — 2024-11-19

### Added
- Population-informed and grouped combination splits
- Multiple split strategies in `src/datasplitting.py`

---

## [0.0.4] — 2024-11-05

### Added
- Train/val/test split with data leakage prevention and temporal consistency
- `.gitignore` updated for outputs and data

---

## [0.0.3] — 2024-10-31

### Added
- Variance, Spearman correlation, and Mann-Whitney U feature selection filters
- SMOTE oversampling integrated into pipeline
- Support for control mice exclusion

### Changed
- Migrated from sklearn balanced datasets to imbalanced-learn pipeline

---

## [0.0.2] — 2024-10-19

### Added
- Cross-validation and hyperparameter search
- TFRecord infrastructure (`tfrecordhandler.py`, `src/create_tfrecords.py`)
- `src/models.py` — model definitions
- `src/train.py` — training utilities
- Initial ML experiments with XGBoost, SVM, Random Forest

---

## [0.0.1] — 2024-10-08

### Added
- Initial dataset preparation and EDA
- `src/radiomics_extractor.py` — PyRadiomics feature extraction (1,218 features)
- `src/preprocessing.py` — scaling, normalisation, encoding utilities
- `src/processing.py` — `Processing` class (CSV loading, imputation, MinMax scaling)
- `config.py` — centralised path and feature name configuration
- First XGBoost model baseline (macro F1 ~0.70–0.78)

---

## [0.0.0] — 2024-10-01

### Added
- Initial commit
- Project structure scaffolding
- Dataset ingestion from raw MRI images

[Unreleased]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.9...v0.1.0
[0.0.9]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/compare/v0.0.0...v0.0.1
[0.0.0]: https://github.com/Javier200107/ML-Strategies-Inflection-Point-GL261-Response-Relapse/commits/0ad0a85
