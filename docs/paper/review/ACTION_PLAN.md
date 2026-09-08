# Plan de Acción — Revisión Mayor Scientific Reports

**Fecha:** 2026-09-09  
**Objetivo:** Preparar la resubmisión respondiendo a todos los comentarios del reviewer.

---

## Visión general

Hay **4 cambios de código** que exigen re-ejecución de experimentos, **3 cambios de texto** que no requieren código, y **1 verificación de PDF**.

Estimación total: ~2–3 semanas de trabajo efectivo.

---

## Bloque 1 — Correcciones de código (afectan resultados)

### A1 · Corregir data leakage: mover feature selection dentro del fold GLOO
**Responde a:** Major Comment 1  
**Prioridad:** CRÍTICA — invalida los resultados actuales si no se corrige  
**Estimación:** 2–3 días

**Qué hay que hacer:**

1. Abrir [notebooks/radiomics/09_final_pipeline_PAPER.ipynb](../../../notebooks/radiomics/09_final_pipeline_PAPER.ipynb)

2. Mover el bloque de feature selection (actualmente ejecutado 1 vez antes del bucle GLOO) **dentro del bucle `for fold, split_data in m_data.items()`**, usando solo `X_train` de ese fold:

```python
# ANTES (incorrecto — opera sobre todo el dataset):
variances = df[features].var()
df.drop(columns=low_variance_features, inplace=True)
corr_matrix = df[m_features].corr(method="spearman")
...

# DESPUÉS (correcto — dentro del fold, solo con train):
for fold, split_data in m_data.items():
    X_train = split_data['X_train'].drop(columns=['m_id'])
    X_test  = split_data['X_test'].drop(columns=['m_id'])
    
    # 1. Filtro varianza (ajustar en train, aplicar a train y test)
    variances = X_train[features].var()
    low_var_feats = variances[variances < 0.01].index.tolist()
    X_train = X_train.drop(columns=low_var_feats)
    X_test  = X_test.drop(columns=low_var_feats, errors='ignore')
    
    # 2. Filtro Spearman (ajustar en train)
    corr_matrix = X_train.corr(method='spearman').abs()
    high_corr_feats = ...  # igual que antes pero sobre X_train
    X_train = X_train.drop(columns=high_corr_feats)
    X_test  = X_test.drop(columns=high_corr_feats, errors='ignore')
    
    # 3. Mann-Whitney U (solo con train)
    for feat in m_features:
        group_0 = X_train[X_train_labels == 0][feat]
        group_1 = X_train[X_train_labels == 1][feat]
        _, p = mannwhitneyu(group_0, group_1, alternative='two-sided')
        if p > 0.05:
            X_train = X_train.drop(columns=[feat])
            X_test  = X_test.drop(columns=[feat], errors='ignore')
    
    # 4. Heurística (eliminar features de filtros)
    ...
    
    # Entrenar XGBoost con X_train filtrado
    ...
```

3. Re-ejecutar el notebook. Los resultados cambiarán (probablemente AUC bajará algo).

4. Guardar nuevos `summary_metrics.csv` y `fold_N/metrics.csv`.

**Nota sobre los modelos DL:** Los pipelines FE y FT no tienen feature selection explícita (EfficientNet extrae features automáticamente), por lo que no tienen este problema de leakage. Solo afecta al pipeline radiomics.

---

### A2 · Añadir métricas animal-level (majority vote)
**Responde a:** Major Comment 2  
**Prioridad:** ALTA  
**Estimación:** 1 día

**Qué hay que hacer:**

Añadir al final del bucle de evaluación (en los 3 notebooks del paper) un bloque que agregue predicciones por ratón:

```python
# Después de poolear all_y_true, all_y_pred, all_y_proba:

# --- Animal-level evaluation (majority vote) ---
import pandas as pd
from scipy import stats

results_df = pd.DataFrame({
    'mouse_id': all_mouse_ids,
    'y_true': all_y_true,
    'y_pred': all_y_pred,
    'y_proba': all_y_proba
})

animal_results = results_df.groupby('mouse_id').agg(
    y_true=('y_true', 'first'),             # la etiqueta es la misma para todos los exámenes
    y_pred_vote=('y_pred', lambda x: stats.mode(x).mode[0]),  # mayoría de votos
    y_proba_mean=('y_proba', 'mean')        # probabilidad media
).reset_index()

# Calcular métricas animal-level
animal_auc  = roc_auc_score(animal_results['y_true'], animal_results['y_proba_mean'])
tn, fp, fn, tp = confusion_matrix(animal_results['y_true'], animal_results['y_pred_vote']).ravel()
animal_sens = tp / (tp + fn)
animal_spec = tn / (tn + fp)
animal_ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
animal_npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
animal_acc  = accuracy_score(animal_results['y_true'], animal_results['y_pred_vote'])

print(f"Animal-level AUC: {animal_auc:.4f}")
print(f"Animal-level Sens: {animal_sens:.4f} | Spec: {animal_spec:.4f}")
print(f"Animal-level PPV: {animal_ppv:.4f} | NPV: {animal_npv:.4f}")
print(f"Animal-level Acc: {animal_acc:.4f}")
```

Aplicar a los 3 notebooks: `09_final_pipeline_PAPER.ipynb`, `01_efficientnet_feature_extraction_PAPER.ipynb`, `02_efficientnet_fine_tuning_PAPER.ipynb`.

Con solo 10 ratones, esperar métricas animal-level por encima de las exam-level (porque el voto agrega información de múltiples exámenes).

---

### A3 · Análisis de rendimiento por ventana temporal (early/mid/late)
**Responde a:** Major Comment 3  
**Prioridad:** ALTA  
**Estimación:** 2–3 días

**Qué hay que hacer:**

1. Recopilar los días de estudio de todos los ratones y dividirlos en tercios (early = días ≤ P33, mid = días P33–P66, late = días > P66) o por valor absoluto (early = días ≤ 14, mid = 15–21, late ≥ 22, ajustar según los datos reales).

2. Para los modelos entrenados (pooled predictions ya disponibles en los CSVs de output), filtrar las predicciones por tercil de tiempo y calcular AUC/Sensitivity por subgrupo:

```python
# Con los resultados ya calculados:
results_df = pd.DataFrame({
    'day': all_days,
    'y_true': all_y_true,
    'y_proba': all_y_proba,
    'y_pred': all_y_pred
})

# Definir ventanas (ajustar umbrales según distribución real de días)
early_mask = results_df['day'] <= 14
mid_mask   = (results_df['day'] > 14) & (results_df['day'] <= 21)
late_mask  = results_df['day'] > 21

for mask, label in [(early_mask, 'Early'), (mid_mask, 'Mid'), (late_mask, 'Late')]:
    subset = results_df[mask]
    if len(subset) > 0 and len(subset['y_true'].unique()) > 1:
        auc  = roc_auc_score(subset['y_true'], subset['y_proba'])
        sens = recall_score(subset['y_true'], subset['y_pred'])
        print(f"{label}: N={len(subset)}, AUC={auc:.3f}, Sens={sens:.3f}")
```

3. Crear un gráfico de AUC / sensitivity en función del día de estudio (o de la ventana temporal) para los 3 modelos → Figura nueva para el paper.

4. Revisar el texto del paper: si el rendimiento es bajo en timepoints tempranos, retirar las afirmaciones de "early prediction" o reformularlas con los datos reales.

---

### A4 · Añadir intervalos de confianza bootstrap a la Tabla 1
**Responde a:** Minor Comment 1  
**Prioridad:** MEDIA  
**Estimación:** 1 día

**Qué hay que hacer:**

```python
from sklearn.utils import resample
import numpy as np

def bootstrap_ci(y_true, y_proba, y_pred, metric_fn, n_bootstrap=10000, ci=0.95):
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(y_true)), random_state=None)
        score = metric_fn(np.array(y_true)[idx], np.array(y_proba)[idx], np.array(y_pred)[idx])
        scores.append(score)
    alpha = (1 - ci) / 2
    return np.percentile(scores, [alpha*100, (1-alpha)*100])

# Ejemplo para AUC:
def auc_fn(yt, yp, _): return roc_auc_score(yt, yp)
ci_low, ci_high = bootstrap_ci(all_y_true, all_y_proba, all_y_pred, auc_fn)
print(f"AUC: {auc_global:.4f} (95% CI: {ci_low:.4f}–{ci_high:.4f})")
```

Aplicar para AUC, Sensitivity, Specificity, PPV, NPV, Accuracy de los 3 modelos.

---

## Bloque 2 — Cambios de texto (sin re-ejecutar código)

### T1 · Corregir "outperforms across all metrics"
**Fichero:** `docs/paper/main.tex`  
**Sección:** Discussion  
**Cambio:** Localizar la frase y sustituir por comparación matizada (ver REVIEWER_RESPONSE.md).

### T2 · Reconocer que DL no modela secuencias longitudinales
**Fichero:** `docs/paper/main.tex`  
**Sección:** Methods (DL subsection) y Discussion  
**Cambio:** Añadir párrafo de limitación (texto en REVIEWER_RESPONSE.md).

### T3 · Discutir PPV bajo
**Fichero:** `docs/paper/main.tex`  
**Sección:** Discussion  
**Cambio:** Añadir párrafo sobre implicaciones clínicas del PPV (texto en REVIEWER_RESPONSE.md).

---

## Bloque 3 — Verificaciones

### V1 · Confirmar que Tablas S3/S4 aparecen en el PDF compilado
**Fichero:** `docs/paper/supplementary.tex`  
**Acción:** Compilar LaTeX y verificar visualmente que las tablas aparecen.

---

## Orden de ejecución recomendado

```
Semana 1:
  [ ] A1 — Corregir feature selection (leakage) → re-ejecutar 09_final_pipeline_PAPER.ipynb
  [ ] A2 — Añadir animal-level metrics a los 3 notebooks
  
Semana 2:
  [ ] A3 — Análisis temporal (early/mid/late)
  [ ] A4 — Bootstrap CIs
  
Semana 3:
  [ ] T1, T2, T3 — Editar main.tex
  [ ] V1 — Compilar PDF y verificar suplementario
  [ ] Redactar carta de respuesta (usar REVIEWER_RESPONSE.md como base)
  [ ] Resubmisión
```

---

## Estado actual de los ficheros de resultados

| Experimento | Output path | Estado |
|-------------|-------------|--------|
| Radiomics+XGB (experimentos 0–8) | `output_final1/hyp3/` | Ejecutado con leakage → **re-ejecutar tras A1** |
| EfficientNet FE+XGB | (en notebook, no guardado a CSV) | Re-ejecutar para obtener animal-level |
| EfficientNet FT | (en notebook, no guardado a CSV) | Re-ejecutar para obtener animal-level |
| Grad-CAM figures | `output_final1/DL/` | OK — no cambian |

---

## Notas importantes

- Con solo **10 ratones** en el conjunto tratado (5 cured, 5 relapsing), los ICs bootstrap serán amplios. Esto es esperable dado el tamaño de la cohorte preclinica y debe discutirse en el texto.
- El análisis temporal (A3) con 10 ratones también tendrá poca potencia estadística por subgrupo. Presentarlo como exploratorio/descriptivo.
- Si tras A1 (corrección de leakage) el AUC del pipeline radiomics cae significativamente, puede ser necesario revisar las conclusiones del paper (y posiblemente el abstract).
