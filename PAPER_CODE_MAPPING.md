# Mapeo: Resultados del Paper → Código del Repositorio

Paper: *"Advanced ML Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"*  
Journal: Scientific Reports (Major Revision recibida)

---

## Tabla 1 del Paper — Los 3 modelos principales

| Modelo | AUC | Sens | Spec | PPV | NPV | Acc |
|--------|-----|------|------|-----|-----|-----|
| Radiomics + XGBoost | 0.7015 | 0.5152 | 0.8000 | 0.6538 | 0.7059 | 0.6864 |
| EfficientNet FE + XGB | 0.8583 | 0.6061 | 0.9600 | 0.9091 | 0.7742 | 0.8046 |
| EfficientNet FT | 0.8511 | 0.9091 | 0.5200 | 0.6522 | 0.8125 | 0.7011 |

Todas las métricas son **a nivel de examen** (exam-level), con validación cruzada **GLOO** (Grouped Leave-One-Out = leave-one-mouse-out).  
*Nota: el reviewer pide clarificar esto y recomienda métricas animal-level.*

---

## 1. Pipeline Radiomics + XGBoost

### Notebook principal
[src/begin/ordered/hyp3 copy/8-baseline_nocontrol_lf_NOsf_var_hc_heuristic_oversample_reduce_again_var_corr.ipynb](src/begin/ordered/hyp3%20copy/8-baseline_nocontrol_lf_NOsf_var_hc_heuristic_oversample_reduce_again_var_corr.ipynb)

### Progresión de experimentos (notebooks 0→8)
Cada notebook añade una capa de refinamiento al pipeline:

| Notebook | Descripción | Cambio clave |
|----------|-------------|--------------|
| `0-baseline_all_features_oversample copy.ipynb` | Baseline con todas las features | 1.218 features, sin selección |
| `1-baseline_all_features_oversample.ipynb` | Ídem | Limpieza menor |
| `2-baseline_nocontrol_oversample.ipynb` | Elimina ratones control | Solo cured vs relapsing |
| `3-baseline_nocontrol_var_hc_oversample.ipynb` | + Filtro varianza | Elimina var < 0.01 |
| `4-baseline_nocontrol_sf_var_hc_oversample.ipynb` | + Filtro correlación Spearman | Elimina \|ρ\| > 0.95 |
| `5-baseline_nocontrol_lf_sf_var_hc_oversample.ipynb` | + Mann-Whitney | Elimina p > 0.05 |
| `6-baseline_nocontrol_lf_sf_var_hc_heuristic_oversample.ipynb` | + Heurística | Elimina features con filtros (wavelet, LoG…) |
| `7-baseline_nocontrol_lf_NOsf_var_hc_heuristic_oversample.ipynb` | Sin Spearman | Prueba sin filtro correlación |
| **`8-baseline_nocontrol_lf_NOsf_var_hc_heuristic_oversample_reduce_again_var_corr.ipynb`** | **← PAPER** | Pipeline definitivo |

### Pipeline de selección de features (dentro de cada fold GLOO)
1. Eliminar features con varianza < 0.01
2. Eliminar features correlacionadas Spearman |ρ| > 0.95
3. Eliminar features no significativas Mann-Whitney U (p > 0.05)
4. Heurística: eliminar features derivadas de filtros (wavelet, LoG, square, etc.)
5. 1.218 features → ~186 features finales

### Parámetros XGBoost (Tabla S3 del paper)
```python
xgb_params = {
    'colsample_bytree': 0.6,
    'gamma': 0.5,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.6
}
```

### Validación cruzada (GLOO)
- Implementada en [src/datasplitting.py](src/datasplitting.py) → método `grouped_combinations_split_with_sampling()`
- test_size = 1 mouse por fold → itera los 10 ratones (5 cured + 5 relapsing)
- Balanceo: SMOTE k_neighbors=5 aplicado **solo al conjunto de train**

### Outputs generados
- `output_final1/hyp3/baseline_no_control_heuristic_lf_NOsf_var_hc_oversample_reduce_again_var_corr/`
  - `grouped_combinations_split_with_sampling/summary_metrics.csv` — métricas promedio
  - `grouped_combinations_split_with_sampling/fold_N/metrics.csv` — métricas por fold
  - `grouped_combinations_split_with_sampling/fold_N/misclassified.csv` — exámenes mal clasificados
  - `grouped_combinations_split_with_sampling/fold_N/well_classified.csv` — exámenes bien clasificados
  - `grouped_combinations_split_with_sampling/top_features.csv` — importancias medias

### Módulos de soporte
- [src/processing.py](src/processing.py) — clase `Processing`: carga CSV, imputa con mediana, escala MinMax
- [src/preprocessing.py](src/preprocessing.py) — funciones de normalización y encoding
- [config.py](config.py) — rutas de datos (`RADIOMICS_FINAL_FEATURES_NEW_FIXED`, etc.)

### Dato de entrada
- `input/features/radiomics_final_features_no_filters.csv` (solo features originales, sin derivadas)
- `input/features/radiomics_final_features_new_fixed.csv` (dataset completo 1.218 features)

---

## 2. Pipeline EfficientNet FE + XGB

### Notebook principal
[src/begin/ordered/hyp3 copy/dl_extractfeatures_img_xgboost.ipynb](src/begin/ordered/hyp3%20copy/dl_extractfeatures_img_xgboost.ipynb)

### Arquitectura
```
Input (256×256×3, ImageNet weights, EfficientNetB0 FROZEN)
→ GlobalAveragePooling2D
→ Dense(256, relu)
→ BatchNormalization
→ Dropout(0.3)
→ [Feature vector de 256 dims]  ← se extraen aquí
→ Dense(1, sigmoid)            ← no se usa para XGBoost
```

### Pipeline
1. Cargar EfficientNetB0 con pesos ImageNet, **todas las capas congeladas** (feature extractor puro)
2. Extraer vector de 256 features por imagen con `feature_extractor.predict()`
3. Leave-One-Group-Out (sklearn `LeaveOneGroupOut`, grupos = `mouse_id`)
4. Por fold: oversample la clase minoritaria (cured) por random resampling hasta igualar mayoritaria
5. Entrenar XGBoost con mismos hiperparámetros que radiomics (n_estimators=100, lr=0.01, max_depth=3, subsample=0.6, colsample=0.6)
6. Poolear predicciones de todos los folds → calcular métricas globales

### Input channel encoding
Las imágenes de entrada son RGB (3 canales), pero son imágenes MRI en escala de grises:
- Canal 0: imagen T2w original
- Canal 1: imagen T2w (repetida)  
- Canal 2: máscara del tumor

### Datos de entrada
- TFRecord: `input/deep_learning/full_ds_fixed_no_control.tfrecord`
  - Solo ratones cured (label=1) y relapsing (label=0), **sin control**
  - Manejado por [tfrecordhandlerDL.py](tfrecordhandlerDL.py) → clase `TFRecordDataHandlerDL`
  - Formato por registro: imagen, máscara, group_name, mouse_id, day_of_study

### Notebooks alternativos del mismo experimento
Otros notebooks `dl_extractfeatures_*.ipynb` en `src/begin/ordered/hyp3 copy/` prueban variantes de combinación de canales (imagen+máscara, imagen+imagen+máscara, etc.):
- `dl_extractfeatures_imgimgmask_xgboost.ipynb` — input: [img, img, mask]
- `dl_extractfeatures_imgmaskmask_xgboost.ipynb` — input: [img, mask, mask]
- `dl_extractfeatures_imgimgimgmask_xgboost_noweights.ipynb` — sin pesos ImageNet
- `dl_extractfeatures_img_xgboost_noweights.ipynb` — sin pesos ImageNet

---

## 3. Pipeline EfficientNet Fine-Tuning (FT)

### Notebook principal
[src/begin/ordered/hyp3 copy/hypotesis1_newds copy_good2 supergood copy 10.ipynb](src/begin/ordered/hyp3%20copy/hypotesis1_newds%20copy_good2%20supergood%20copy%2010.ipynb)

### Diferencia clave vs FE
En FT, las **últimas 50 capas** de EfficientNetB0 son entrenables:
```python
for layer in base_model.layers[:-50]:
    layer.trainable = False  # Solo congelar las primeras capas
# → últimas 50 capas se fine-tunean por fold
```

### Pipeline
1. Por cada fold GLOO:
   - Construir modelo nuevo con `build_model()` (pesos ImageNet frescos)
   - Fine-tune durante 20 epochs (Adam lr=1e-4, batch_size=32, binary_crossentropy)
   - Predecir en el ratón de test
2. Poolear predicciones → métricas globales finales

### Input channel encoding
- Canal 0: imagen T2w
- Canal 1: imagen T2w (repetida)
- Canal 2: máscara del tumor (misma estrategia que FE)

### Notebooks de versiones anteriores (src/notebooks/dl/)
- `hypotesis1OK.ipynb` — versión con VGG16 (no aparece en el paper final)
- `hypotesis1OK copy.ipynb` — variante VGG16
- `hypotesis1_newds copy_good2.ipynb` — versión intermedia con EfficientNet
- Notebooks numerados `copy 9`, `copy 10`, `copy 11` — iteraciones del FT definitivo

---

## 4. Grad-CAM (Figura S4 del Suplementario)

### Código
- [src/notebooks/dl/gradcam.py](src/notebooks/dl/gradcam.py) — clase `GradCAM`
- Implementado también inline en notebooks DL

### Outputs generados (usados en el paper)
Directorio: `output_final1/DL/`

| Archivo | Descripción |
|---------|-------------|
| `1263_d11_mri.png` | Imagen MRI ratón C1263, día 11 |
| `1263_d11_heat.png` | Heatmap Grad-CAM, día 11 |
| `1263_d11_gradcam.png` | Superposición Grad-CAM, día 11 |
| `1263_d14_*.png` | Ídem día 14 |
| `1263_d17_*.png` | Ídem día 17 |
| `1263_d19_*.png` | Ídem día 19 |
| `1382.png` | Imagen ratón C1382 |
| `1382_grad.png` | Grad-CAM ratón C1382 |

Ratón C1263: ratón **cured** analizado longitudinalmente (días 11→19) para mostrar evolución de la atención del modelo.

---

## 5. Datos y preprocesamiento

### Dataset original
- 18 ratones C57BL/6 con GL261 glioblastoma tratados con TMZ
  - 8 control (excluidos del análisis principal)
  - 5 relapsing → label = 0
  - 5 cured → label = 1
- 169 exámenes MRI T2w en total (variable por ratón)
- IDs de ratones en el código: 1263, 1264, 1270, 1276, 1281, 1284, 1285, 1380, 1382, 1383
- IDs control (excluidos): 1258, 1260, 1261, 1297, 1299, 1359, 1360, 1361

### Extracción de features radiómicas
- [src/radiomics_extractor.py](src/radiomics_extractor.py) — usa PyRadiomics
- 1.218 features totales (originales + wavelet + LoG + square + logarithm + exponential)

### Datos de imágenes para DL
- [tfrecordhandlerDL.py](tfrecordhandlerDL.py) — `TFRecordDataHandlerDL`: lee TFRecord con (imagen, máscara, label, mouse_id, day)
- [tfrecordimagehandler.py](tfrecordimagehandler.py) — handler alternativo para solo imagen

---

## 6. Estructura de directorios de outputs

```
output_final1/
├── DL/                          ← Imágenes Grad-CAM (Fig. S4)
│   ├── 1263_d{11,14,17,19}_*.png
│   └── 1382*.png
└── hyp3/
    ├── baseline_all_features/                          ← Exp 0 (todas features)
    ├── baseline_all_features_oversample/               ← Exp 1
    ├── baseline_no_control_oversample/                 ← Exp 2
    ├── baseline_no_control_var_hc_oversample/          ← Exp 3
    ├── baseline_no_control_sf_var_hc_oversample/       ← Exp 4
    ├── baseline_no_control_lf_sf_var_hc_oversample/   ← Exp 5
    ├── baseline_no_control_heuristic_lf_sf_var_hc_oversample/ ← Exp 6
    ├── baseline_no_control_heuristic_lf_NOsf_var_hc_oversample/ ← Exp 7
    ├── baseline_no_control_heuristic_lf_NOsf_var_hc_oversample_reduce_again_var_corr/  ← **PAPER Exp 8**
    ├── baseline_no_control_heuristic_lf_NOsf_var_hc_undersample_reduce_again_var_corr/ ← Exp 9
    ├── RF/                                             ← Random Forest baseline
    └── RF_baseline_no_control_heuristic_.../           ← RF final

output_final/
└── hyp1/
    └── imgs/                    ← Expanding Window (TFM, no en el paper)
```

---

## 7. Revisión Mayor — Cuestiones pendientes de código

El reviewer señala varios puntos que requieren verificación en el código:

### R1: Posible data leakage en selección de features
**Pregunta**: ¿La selección de features (varianza + correlación + Mann-Whitney) se hace dentro de cada fold GLOO o sobre todo el dataset?

**En el código actual** ([4-baseline_nocontrol_sf_var_hc_oversample.ipynb](src/begin/ordered/hyp3%20copy/4-baseline_nocontrol_sf_var_hc_oversample.ipynb)):
```python
# ← Este código se ejecuta ANTES de la división GLOO
variances = df[features].var()
df.drop(columns=low_variance_features, ...)
corr_matrix = df[m_features].corr(method="spearman")
df.drop(columns=high_corr_features, ...)
mannwhitneyu(group_0, group_1, ...)
```
La selección ocurre **fuera del bucle GLOO**, sobre todo el dataset → **potencial leakage**.

### R2: Métricas exam-level vs animal-level
Las métricas en `summary_metrics.csv` y en la Tabla 1 del paper son exam-level (cada examen MRI cuenta como una muestra independiente). El reviewer recomienda añadir métricas animal-level (mayoría de votos por ratón).

### R3: Features importantes (Figuras S1, S5)
Las features más importantes identificadas en el paper (Elongation, RunEntropy, DependenceEntropy, SRHGLE, HGRLE) vienen de:
- `output_final1/hyp3/.../grouped_combinations_split_with_sampling/top_features.csv`
- Visualizadas con boxplots (Fig S1) y análisis temporal (Fig S3)

---

## 8. Resumen ejecutivo del flujo completo

```
Datos MRI (input/dataset/)
    ↓ src/radiomics_extractor.py
Radiomics CSV (input/features/*.csv) ─────────────────────────────────────┐
    ↓ src/processing.py (Processing.read())                                │
DataFrame preprocesado (1.218 features)                                   │
    ↓ Notebooks hyp3/0-8 (selección features)                             │
~186 features seleccionadas                                               │
    ↓ src/datasplitting.py (grouped_combinations_split_with_sampling)     │
GLOO folds (10 ratones, 1 test por fold)                                  │
    ↓ SMOTE + XGBoost (xgb_params fijos)                                  │
Predicciones pooled → Tabla 1 fila 1 (AUC=0.7015)                       │
                                                                           │
Datos MRI (input/deep_learning/full_ds_fixed_no_control.tfrecord) ←──────┘
    ↓ tfrecordhandlerDL.py (TFRecordDataHandlerDL)
Dataset imágenes (imagen T2 + máscara, 10 ratones)
    ↓ EfficientNetB0(ImageNet, frozen)
    ↓ Feature extraction (256-dim) + LOGO + XGBoost
Predicciones → Tabla 1 fila 2 (AUC=0.8583)
    ↓ EfficientNetB0(ImageNet, last 50 layers trainable, fine-tune per fold)
Predicciones → Tabla 1 fila 3 (AUC=0.8511, Sens=0.9091)
    ↓ Grad-CAM (top conv layer)
output_final1/DL/ (Figura S4)
```
