"""Compute NPV bootstrap CIs from saved fold predictions and save to CSV."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

SEED = 42
N_BOOT = 10000
np.random.seed(SEED)

def load_predictions(base_dir):
    dfs = []
    for fold_dir in sorted(Path(base_dir).glob('fold_*')):
        p = fold_dir / 'predictions.csv'
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else None

def npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan

def bootstrap_ci(y_true, y_pred, n_boot=N_BOOT):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(npv(yt, yp))
    return np.percentile(scores, [2.5, 97.5])

models = {
    'FE+XGB': 'outputs/paper_revision/deep_learning/01_efficientnet_FE_XGB',
    'FT':     'outputs/paper_revision/deep_learning/02_efficientnet_FT',
}

for name, path in models.items():
    df = load_predictions(path)
    if df is None:
        print(f'{name}: no predictions found'); continue
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    npv_point = npv(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    # Append to existing bootstrap_cis.csv or create new
    out_csv = Path(path) / 'bootstrap_cis.csv'
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        if 'NPV' not in existing['metric'].values:
            npv_row = pd.DataFrame([{
                'metric': 'NPV',
                'point_estimate': npv_point,
                'ci_lower_2_5': lo,
                'ci_upper_97_5': hi,
                'n_bootstrap_valid': N_BOOT,
            }])
            pd.concat([existing, npv_row], ignore_index=True).to_csv(out_csv, index=False)
            print(f'{name}: NPV = {npv_point:.4f} [{lo:.4f}, {hi:.4f}] -> saved to {out_csv}')
        else:
            print(f'{name}: NPV already in {out_csv}')
    else:
        print(f'{name}: {out_csv} not found, skipping')
