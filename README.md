# ML Strategies for Predicting Therapy Response in Preclinical Glioblastoma

Code repository for the paper:  
**"Advanced ML Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"**  
*J. González Béjar, A. P. Candiota, A. Vellido — Scientific Reports (under revision)*

---

## Repository Structure

```
├── config.py                        # Global paths and feature name constants
├── PAPER_CODE_MAPPING.md            # Map of paper results → code files
│
├── notebooks/
│   ├── radiomics/
│   │   ├── 01_baseline_all_features.ipynb
│   │   ├── 02_baseline_all_features_v2.ipynb
│   │   ├── 03_no_control.ipynb
│   │   ├── 04_filter_variance.ipynb
│   │   ├── 05_filter_variance_spearman.ipynb
│   │   ├── 06_filter_variance_spearman_mannwhitney.ipynb
│   │   ├── 07_filter_heuristic_with_spearman.ipynb
│   │   ├── 08_filter_heuristic_no_spearman.ipynb
│   │   ├── 09_final_pipeline_PAPER.ipynb  ← PAPER Table 1, row 1
│   │   └── expanding_window_TFM.ipynb     ← TFM expanding window experiment
│   │
│   └── deep_learning/
│       ├── 01_efficientnet_feature_extraction_PAPER.ipynb  ← PAPER Table 1, row 2
│       ├── 02_efficientnet_fine_tuning_PAPER.ipynb         ← PAPER Table 1, row 3
│       └── gradcam.py                                       ← Grad-CAM implementation
│
├── src/
│   ├── processing.py               # Radiomics CSV loading and preprocessing
│   ├── preprocessing.py            # Scaling, normalisation, encoding utilities
│   ├── datasplitting.py            # GLOO cross-validation splitter
│   ├── datasplitting_for_images.py # GLOO splitter for image datasets
│   ├── balance.py                  # SMOTE / undersampling helpers
│   ├── radiomics_extractor.py      # PyRadiomics feature extraction
│   ├── create_tfrecords.py         # Build TFRecord files from raw images
│   ├── tfrecordhandlerDL.py        # TFRecord reader for DL pipeline
│   ├── tfrecordhandler.py          # TFRecord reader (alternative)
│   ├── models.py                   # Model definitions
│   └── train.py                    # Training utilities
│
├── docs/
│   ├── paper/
│   │   ├── main.tex                # Paper manuscript
│   │   ├── supplementary.tex       # Supplementary material
│   │   ├── refs.bib                # Bibliography
│   │   ├── images/                 # Figures used in the paper
│   │   └── review/                 # Reviewer comments
│   └── tfm.tex                     # Master's thesis (UPC)
│
├── input/                           # Data (not versioned)
│   ├── features/                    # Radiomics CSV files (1,218 features)
│   └── deep_learning/               # TFRecord datasets (full_ds_fixed_no_control.tfrecord)
│
├── outputs/                         # Experiment results (not versioned)
│   ├── radiomics/
│   │   ├── gloo/
│   │   │   ├── 01_baseline_all_features/      # Exp 01 — all 1,218 features
│   │   │   ├── 02_baseline_all_features_v2/   # Exp 02
│   │   │   ├── 03_no_control/                 # Exp 03 — drop control mice
│   │   │   ├── 04_filter_variance/            # Exp 04 — + variance filter
│   │   │   ├── 05_filter_variance_spearman/   # Exp 05 — + Spearman filter
│   │   │   ├── 06_filter_mannwhitney/         # Exp 06 — + Mann-Whitney filter
│   │   │   ├── 07_filter_heuristic_with_spearman/ # Exp 07
│   │   │   ├── 08_filter_heuristic_no_spearman/   # Exp 08
│   │   │   ├── 09_final_pipeline_PAPER/       # ← PAPER Table 1, row 1
│   │   │   ├── 09b_undersample_variant/       # Exp 09 (alternative, not in paper)
│   │   │   ├── RF_baseline/                   # Random Forest baseline
│   │   │   └── RF_final_pipeline/             # RF with same feature selection
│   │   └── expanding_window/                  # TFM expanding window results
│   ├── deep_learning/
│   │   └── gradcam/                           # Grad-CAM figures (mouse C1263, days 11-19)
│   └── figures/                               # Feature importance plots, boxplots, t-SNE/UMAP
│
│   Each GLOO experiment subfolder contains:
│     grouped_combinations_split_with_sampling/
│       ├── summary_metrics.csv          # Overall mean metrics
│       ├── top_features.csv             # Mean feature importances
│       └── fold_N/
│           ├── metrics.csv              # Per-fold metrics
│           ├── misclassified.csv        # Wrong predictions with day/mouse info
│           └── well_classified.csv      # Correct predictions
│
└── archive/                         # Deprecated / exploratory code
    ├── notebooks_experimental/
    └── data_intermediates/
```

## Paper Models Summary

| Model | AUC | Sensitivity | Specificity | Accuracy |
|-------|-----|-------------|-------------|----------|
| Radiomics + XGBoost (GLOO) | 0.7015 | 0.5152 | 0.8000 | 0.6864 |
| EfficientNet FE + XGB (GLOO) | 0.8583 | 0.6061 | 0.9600 | 0.8046 |
| EfficientNet Fine-tuned (GLOO) | 0.8511 | 0.9091 | 0.5200 | 0.7011 |

*All metrics are exam-level. GLOO = Grouped Leave-One-Out (leave-one-mouse-out) cross-validation.*

## Setup with uv

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### First-time setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# or on Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create virtual environment and install dependencies
uv sync

# 3. Install TensorFlow with GPU support (Windows + CUDA 11.x, Python 3.10 required)
uv pip install tensorflow==2.10.0
# or CPU-only:
uv pip install tensorflow-cpu==2.10.0

# 4. Register kernel for Jupyter
uv run python -m ipykernel install --user --name gl261 --display-name "GL261 (uv)"
```

### Daily use

```bash
# Run a notebook
uv run jupyter notebook notebooks/radiomics/09_final_pipeline_PAPER.ipynb

# Run a script
uv run python src/radiomics_extractor.py
```

### Note on GPU / corporate network

If `uv sync` fails to download Python 3.10 (corporate firewall), use an existing Python installation:

```bash
# Point uv to the existing conda environment Python
uv venv --python "C:\Users\xavie\miniconda3\envs\tfm_new\python.exe"
uv pip install -e .
```

Alternatively, install packages directly via uv into the conda env:
```bash
# From within the conda env
conda activate tfm_new
uv pip install -e .          # installs the project in editable mode
```

Key dependencies: TensorFlow 2.10, XGBoost, scikit-learn, PyRadiomics, imbalanced-learn.

Run notebooks from the repository root so that `src/` imports resolve correctly.
