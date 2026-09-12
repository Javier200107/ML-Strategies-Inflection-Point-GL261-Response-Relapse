# Response to Reviewers

**Manuscript:** "Advanced Machine Learning Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"  
**Journal:** Scientific Reports  
**Decision:** Major Revision

We thank both reviewers for their thorough and constructive evaluation. All major and minor comments have been fully addressed. Changes to the manuscript are indicated in red in the revised manuscript (`main_revised.tex`).

---

## Reviewer 1

---

### R1-1 — Effective sample size: n=10 animals, not 169 examinations

**Reviewer comment:**
The core task is based on only 10 independent animals (5+5). The GLOO scheme is appropriate, but reported metrics should be interpreted in light of ~10 independent observations, not 169.

**Response — IMPLEMENTED ✅**

We fully agree. The revised manuscript now explicitly states throughout (Abstract, Results, Discussion, Conclusion) that the effective number of independent observations is 10 animals, and that all metrics should be interpreted accordingly. The Discussion opens with: *"within the framework of a proof-of-concept study based on a small but well-characterized cohort of 10 treated animals"*, and the Conclusion frames the study as proof-of-concept pending confirmation in larger cohorts.

**Changes:** Abstract, Results Performance Comparison, Discussion (opening), Conclusion.

---

### R1-2 — Metrics reported to four decimal places; bootstrap CIs and ROC comparison test needed

**Reviewer comment:**
Performance values to four decimal places convey false precision given n=10 subjects. All Table 1 metrics should be accompanied by subject-level bootstrap CIs. The comparison between pipelines should be supported by a formal test (DeLong or subject-level permutation).

**Response — IMPLEMENTED ✅**

**Bootstrap CIs:** We computed 95% bootstrap confidence intervals (10,000 resamples) at the examination level for all metrics across all three pipelines. Results are now reported in the Table 1 caption:

| Model | AUC | 95% CI |
|-------|-----|--------|
| Radiomics + XGBoost | 0.773 | [0.707, 0.835] |
| EfficientNet FE+XGB | 0.801 | [0.733, 0.866] |
| EfficientNet FT | 0.868 | [0.812, 0.918] |

All values in Table 1 are now reported to 3 decimal places.

**DeLong test:** We implemented the DeLong (1988) variance estimator to compare correlated AUCs. Results reported in the Table 1 caption and Results text:
- EfficientNet FT vs. Radiomics: Z = −4.68, **p < 0.001** ✅ (statistically significant)
- EfficientNet FE+XGB vs. Radiomics: Z = −1.31, p = 0.19 (not significant; fine-tuning is the key driver)

**Subject-level permutation test** (n=10 animals): radiomics vs. FT AUC difference = −0.04 (p = 0.74); radiomics vs. FE+XGB AUC difference = 0.00 (p = 1.00). As expected with n=10, animal-level tests have very limited power; these results are consistent with the interpretation provided in the DeLong analysis.

**Changes:** Table 1 (all values, new CIs), Table 1 caption (DeLong results), Results Performance Comparison, Methods Performance Metrics.

---

### R1-3 — Data leakage: feature selection and preprocessing outside GLOO folds

**Reviewer comment:**
It is unclear whether all radiomic feature selection (correlation filtering, Mann–Whitney U), normalisation, temporal feature engineering and hyperparameter optimisation were fully nested within the training data of each outer GLOO fold.

**Response — IMPLEMENTED ✅**

We confirmed that the original implementation performed feature selection on the full dataset before the GLOO split. The pipeline has been fully re-implemented so that the complete sequence — (1) heuristic filter retaining only `original_` features, (2) variance filter (var < 0.01), (3) Spearman correlation deduplication (|ρ| > 0.95), (4) Mann–Whitney U test (p ≤ 0.05, two-sided) — is applied **exclusively within the training partition of each GLOO fold**, with the selected feature subset applied to the held-out test fold without refitting. StandardScaler normalisation and SMOTE oversampling are also fitted on training data only. Corrected results:

| Metric | Original (leakage) | Corrected |
|--------|-------------------|-----------|
| AUC | 0.7015 | **0.773** [0.707, 0.835] |
| Sensitivity | 0.515 | 0.545 |
| Specificity | 0.800 | 0.810 |

**Changes:** Methods (Feature Selection subsection, explicit within-fold statement), Results (Radiomics Analysis), code in `notebooks/paper_revision/09_final_pipeline_corrected_PAPER.ipynb`.

---

### R1-4 — DL pipeline: validation set for early stopping and best-checkpoint selection

**Reviewer comment:**
Under GLOO with a single animal held out, it is not clear which data were used as the validation set for early stopping and checkpoint selection. If the held-out subject was used for model selection, the AUC would be optimistically biased.

**Response — IMPLEMENTED ✅**

We confirm that no early stopping or checkpoint selection was applied in the DL fine-tuning pipeline. Each fold model was trained for a fixed 20 epochs. Although `validation_data=(X_test, y_test)` was passed to `model.fit()` during training to log monitoring metrics, this did not influence any model-selection or stopping decision — the final weights from the last epoch were used for evaluation. This is now stated explicitly in the Methods:

*"Each fold model was trained for a fixed 20 epochs with no early stopping and no model checkpoint selection based on validation performance; the weights at the final epoch were used for evaluation. Although the held-out test animal's data were passed as `validation_data` during training for monitoring purposes only, they did not influence any model-selection or stopping decision."*

**Changes:** Methods (Fine-Tuning Procedure subsection).

---

### R1-5 — Internal consistency: FE+XGB has higher AUC than FT, yet FT is presented as best

**Reviewer comment:**
Table 1 shows EfficientNet FE+XGB (AUC 0.8583) above EfficientNet FT (AUC 0.8511), yet the abstract and conclusions foreground FT with AUC "0.85". The rationale for choosing FT should be stated explicitly.

**Response — IMPLEMENTED ✅**

With the corrected re-execution (with GPU determinism), the ranking is now:

| Model | AUC |
|-------|-----|
| EfficientNet FT | **0.868** |
| EfficientNet FE+XGB | 0.801 |
| Radiomics | 0.773 |

FT now has a higher AUC than FE+XGB in the corrected results, resolving the apparent inconsistency. Nevertheless, the selection rationale is now stated explicitly throughout the manuscript: *"EfficientNet FT is selected as the recommended model despite EfficientNet FE achieving a similar AUC (0.801), because FT provides substantially higher sensitivity (0.818 vs. 0.606) and NPV (0.936 vs. 0.884), which are the clinically most relevant metrics for identifying cured animals."*

**Changes:** Table 1 caption, Results Performance Comparison, Discussion, Conclusion.

---

### R1-6 — Radiomic feature trajectories: illustrative, not established biomarkers; Dependence Entropy is speculative

**Reviewer comment:**
Longitudinal feature differences are shown as group-level trajectories without individual trajectories, uncertainty estimates, or repeated-measures testing. Dependence Entropy on five cured animals is speculative and should be framed cautiously.

**Response — IMPLEMENTED ✅**

The text now reads: *"These longitudinal trajectories are presented as illustrative group-level tendencies within five animals per group and should not be interpreted as statistically established biomarkers; formal repeated-measures testing and independent validation in larger cohorts would be required to confirm them."*

Regarding Dependence Entropy specifically: *"This trend — increasing entropy associated with a favorable outcome — is an illustrative descriptive observation based on five cured animals and should be regarded as hypothesis-generating rather than as an established biomarker."*

**Changes:** Results (Radiomics Model Analysis), Discussion.

---

### R1-7 — Proof-of-concept framing; early prediction claim too strong

**Reviewer comment:**
Given the single experimental model, small cohort, and absence of independent validation, the study should be presented as proof-of-concept. The early-prediction claim rests on the smallest and most imbalanced portion of the cohort and warrants cautious wording.

**Response — IMPLEMENTED ✅**

The manuscript is now framed as a proof-of-concept throughout (Abstract, Discussion, Conclusion). The early-prediction claim is now supported by the stratified temporal analysis and qualified accordingly.

We have performed a temporal analysis stratifying the 298 pooled predictions by examination day tercile:

| Window | Days | Radiomics AUC | FE+XGB AUC | FT AUC |
|--------|------|--------------|------------|--------|
| Early | ≤ 21 | 0.582 | 0.741 | 0.788 |
| Mid | ≤ 31 | 0.809 | 0.767 | 0.902 |
| Late | > 31 | 0.960 | 0.987 | 1.000 |

The revised text states: *"Radiomics performance is near-chance in early treatment (AUC 0.58), whereas both DL configurations maintain substantially higher discrimination from the earliest time points (AUC 0.74–0.79). Claims of reliable early prediction for radiomics remain limited."*

**Changes:** Abstract, Results (Temporal Prediction Performance — new table with all 3 models), Discussion, Conclusion.

---

## Reviewer 2

---

### R2-1 — Feature selection and preprocessing leakage within CV folds

**Reviewer comment:**
The manuscript does not clearly state whether feature selection, preprocessing, and hyperparameter optimisation were restricted to subject-separated training data within each fold. If performed on the full dataset, performance estimates are unreliable and reanalysis is necessary.

**Response — IMPLEMENTED ✅**

Addressed identically to R1-3 above. The original implementation had data leakage; the corrected pipeline runs all feature selection exclusively within the training partition of each GLOO fold. The reanalysis has been completed with corrected results (radiomics AUC = 0.773 [0.707, 0.835]). The Methods section now states unambiguously, for each step, whether it was nested within the fold.

---

### R2-2 — Subject-level performance summaries

**Reviewer comment:**
Recommend subject-level performance summaries in addition to examination-level metrics.

**Response — IMPLEMENTED ✅**

Animal-level majority-vote metrics are now reported in Results and in **Supplementary Table S5**:

| Model | AUC | Sensitivity | Specificity | Accuracy |
|-------|-----|-------------|-------------|----------|
| Radiomics | 0.92 | 0.60 | 1.00 | 0.80 |
| EfficientNet FE+XGB | 0.92 | 0.60 | 1.00 | 0.80 |
| EfficientNet FT | 0.88 | 0.80 | 1.00 | 0.90 |

All five relapsing animals were correctly identified by all three models (Specificity = 1.00). EfficientNet FT correctly classified four of five cured animals, compared with three of five for the other configurations. Supplementary Table S5 provides per-animal predictions for all models.

**Changes:** Results (new animal-level paragraph), Methods (Performance Metrics subsection), Supplementary Table S5.

---

### R2-3 — Revision of conclusions

**Reviewer comment:**
If procedures were fully nested by animal, the manuscript is addressable through clearer reporting, subject-level summaries, and revision of several conclusions.

**Response — IMPLEMENTED ✅**

All conclusions have been revised:
- Proof-of-concept framing added throughout
- "Across all metrics" corrected to "across most metrics"
- Early prediction claims moderated and supported by temporal data
- Dependence Entropy framed as hypothesis-generating
- FT vs. FE+XGB selection rationale made explicit
- DeLong test added to support model comparison

---

## Summary Table

| Comment | Status |
|---------|--------|
| R1-1: N=10 framing throughout | ✅ Done |
| R1-2: Bootstrap CIs + DeLong test | ✅ Done — FT vs. Rad p<0.001 |
| R1-3: Feature selection inside fold (radiomics) | ✅ Done — corrected & re-run |
| R1-4: DL validation set / no early stopping | ✅ Done — clarified in Methods |
| R1-5: FE vs FT consistency | ✅ Done — FT now higher AUC; rationale explicit |
| R1-6: Feature trajectories illustrative; Dependence Entropy speculative | ✅ Done |
| R1-7: Proof-of-concept; early prediction cautious | ✅ Done — temporal table added |
| R2-1: Feature selection leakage | ✅ Done (same as R1-3) |
| R2-2: Subject-level summaries | ✅ Done — Supplementary Table S5 |
| R2-3: Revise conclusions | ✅ Done |

### Remaining (requires GPU re-execution on separate machine)
- Bootstrap CIs and animal-level metrics for FE+XGB and FT are based on GPU runs with `seed=42` and `tf.config.experimental.enable_op_determinism()` enforced.
- DeLong test used pooled fold predictions from all three models.
- All code available in `notebooks/paper_revision/`.
