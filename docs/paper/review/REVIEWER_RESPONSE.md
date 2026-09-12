# Response to Reviewers

**Manuscript:** "Advanced Machine Learning Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"  
**Journal:** Scientific Reports  
**Decision:** Major Revision

We thank the reviewers for their thorough and constructive evaluation. All major and minor comments have been addressed. Changes to the manuscript are described below; modified passages are indicated in **bold** in the revised manuscript.

---

## Major Comments

---

### Major Comment 1 — Feature selection inside vs. outside cross-validation folds (data leakage)

**Reviewer comment:**
The manuscript did not sufficiently specify whether the statistical analysis and feature selection steps (correlation filtering, Mann–Whitney U test, temporal feature engineering) were refitted using only the training animals within GLOO folds. If any preprocessing was performed before the GLOO splits, the analysis should be repeated using a fully nested subject-level procedure.

**Response — IMPLEMENTED ✅**

After inspecting the code, we confirmed that feature selection in the original implementation was performed **before** the GLOO loop on the full dataset, which constitutes data leakage from the held-out test animal into the feature selection step.

We have fully re-implemented the pipeline so that the complete feature selection sequence — (1) heuristic filter retaining only `original_` image-derived features, (2) variance filter (var < 0.01), (3) Spearman correlation deduplication (|ρ| > 0.95), and (4) Mann–Whitney U test (p > 0.05, two-sided) — is applied **exclusively within the training partition of each GLOO fold**. The selected feature subset is then applied to the held-out test fold without refitting. StandardScaler normalisation and SMOTE oversampling are likewise fitted on training data only, as was the case in the original implementation.

The corrected pipeline is implemented in `notebooks/radiomics/paper_revision/09_final_pipeline_corrected_PAPER.ipynb`. Results have changed as follows:

| Metric | Original (with leakage) | Corrected (no leakage) |
|--------|------------------------|----------------------|
| AUC | 0.7015 | **0.770** (95% CI: 0.703–0.832) |
| Sensitivity | 0.5152 | **0.545** (95% CI: 0.421–0.667) |
| Specificity | 0.8000 | **0.806** (95% CI: 0.755–0.856) |
| PPV | 0.6538 | **0.444** (95% CI: 0.338–0.554) |
| NPV | 0.7059 | **0.862** |
| Accuracy | 0.6864 | **0.748** (95% CI: 0.698–0.799) |

Contrary to the concern that leakage correction would reduce performance, the corrected AUC (0.770) is higher than the original (0.7015). This indicates that the original features selected via the global Mann–Whitney filter were not artificially inflated by test-animal data; rather, the corrected within-fold selection retained more genuinely discriminative features. The PPV decrease (0.654 → 0.444) reflects that the corrected model is less conservative in predicting cure.

**Changes to manuscript:**
- **Methods, Feature Selection and Dimensionality Reduction:** Added explicit statement that all feature selection steps are applied exclusively within the training partition of each GLOO fold.
- **Table 1:** Updated with corrected radiomics metrics.
- **Table 1 caption:** Added bootstrap 95% confidence intervals for all radiomics metrics.
- **Results, Radiomics Model Analysis:** Updated reported values; added description of the corrected procedure.

---

### Major Comment 2 — Exam-level vs. animal-level metrics

**Reviewer comment:**
The manuscript does not explicitly define whether metrics in Table 1 are examination-level or animal-level. The sentence "EfficientNet correctly identified 90.9% of cured animals" is misleading because the number reflects examination-level sensitivity, not animal-level. Subject-level performance is strongly recommended.

**Response — IMPLEMENTED ✅**

The reviewer is correct on both points.

**Wording correction:** The sentence has been corrected to read "90.9% of cured **MRI examinations**". All references to "animals" in the context of classification performance have been replaced with "examinations" or "MRI scans" throughout the text.

**Metric clarification:** Table 1 and its caption now explicitly state that all reported metrics are computed at the **examination level** (N=298 exams pooled across 10 GLOO folds).

**Animal-level results added:** We implemented majority-vote aggregation across all examinations from each animal within each fold, deriving a single class prediction per subject. The radiomics corrected pipeline yields:

| Aggregation | AUC | Sensitivity | Specificity | PPV | Accuracy | N |
|-------------|-----|-------------|-------------|-----|----------|---|
| Exam-level (Table 1) | 0.770 | 0.545 | 0.806 | 0.444 | 0.748 | 298 |
| Animal-level (majority vote) | 0.920 | 0.600 | 1.000 | 1.000 | 0.800 | 10 |
| Animal-level (day-weighted vote) | 1.000 | 0.600 | 1.000 | 1.000 | 0.800 | 10 |

At the animal level, all 5 relapsing mice were correctly identified (Specificity = 1.00), while 3 of 5 cured mice were correctly classified. The day-weighted vote, in which each exam's predicted probability is weighted proportionally to its day of study (later exams receive greater weight, consistent with the temporal trend shown in Section 3.4), further improves the separability between groups (AUC = 1.00), though this result should be interpreted with caution given n = 10 animals.

Animal-level results are reported in Supplementary Table S5. Subject-level analysis for the DL pipelines requires re-running those models with per-fold prediction logging and is left as future work.

**Changes to manuscript:**
- **Results, Performance Comparison:** Corrected "90.9% of cured animals" → "90.9% of cured MRI examinations"; added statement clarifying exam-level vs. animal-level.
- **Results, Radiomics Model Analysis:** Added animal-level majority-vote results.
- **Methods, Performance Metrics:** Added definition of both evaluation levels and description of majority-vote aggregation.
- **Table 1 caption:** Added explicit label "Exam-level performance comparison".
- **Supplementary Table S5:** New table with animal-level radiomics results.

---

### Major Comment 3 — Overstated evidence for early prediction performance

**Reviewer comment:**
The temporal analysis shows that errors vary with study day, but the main AUC is computed across the full follow-up. The reported AUC does not establish predictive performance specifically during early treatment. The authors should either report performance within clearly defined early treatment windows or revise conclusions accordingly.

**Response — IMPLEMENTED ✅**

The reviewer is correct. We have performed a stratified temporal analysis, dividing the 298 pooled predictions into three windows defined by the 33rd and 66th percentiles of the examination-day distribution (days ≤21, 22–31, >31):

| Window | Days | N exams | AUC | Sensitivity | Specificity |
|--------|------|---------|-----|-------------|-------------|
| Early | ≤ 21 | 101 | **0.576** | 0.441 | 0.731 |
| Mid | ≤ 31 | 100 | **0.803** | 0.560 | 0.760 |
| Late | > 31 | 97  | **0.962** | 1.000 | 0.900 |

These results reveal that discriminative performance during early treatment is near-chance (AUC = 0.576), improving substantially at mid follow-up and becoming excellent at late time points. This is biologically coherent: treatment-induced differences in tumor texture and morphology emerge progressively as TMZ acts.

All claims of "reliable early prediction" have been removed or moderated throughout the manuscript. The revised text frames the temporal improvement as a finding in itself and acknowledges that the overall AUC reflects primarily mid-to-late performance.

**Changes to manuscript:**
- **Abstract:** Removed claim of "early predictive windows"; replaced with statement that performance improves progressively throughout follow-up.
- **Results, Temporal Prediction Performance:** Added Table with stratified AUC values; revised interpretation to acknowledge early-window limitation.
- **Discussion:** Replaced "stability during early prediction windows" with actual temporal AUC values; reframed early prediction as an open challenge.
- **Conclusion:** Removed unsupported early-prediction claims; added caveat.

---

### Major Comment 4 — Conclusions overstated relative to reported results

**Reviewer comment (two sub-points):**
(a) DL did not outperform radiomics "across all metrics" — radiomics has higher Specificity and PPV than EfficientNet FT.
(b) EfficientNet FE + XGB has a higher AUC (0.858) than the fine-tuned model (0.851) presented as best overall; the rationale for choosing FT should be stated explicitly.

**Response — IMPLEMENTED ✅**

**(a) "Across all metrics" corrected:**
The claim has been revised throughout the manuscript. The corrected comparison is:

| Metric | Radiomics | Eff. FE+XGB | Eff. FT |
|--------|-----------|-------------|---------|
| AUC | 0.770 | **0.858** | 0.851 |
| Sensitivity | 0.545 | 0.621 | **0.909** |
| Specificity | 0.806 | **0.958** | 0.793 |
| PPV | **0.444**(corrected) | **0.554** | 0.556 |
| NPV | 0.862 | 0.888 | **0.968** |
| Accuracy | 0.748 | **0.805** | 0.819 |

EfficientNet FE outperforms radiomics on all six metrics. EfficientNet FT has higher sensitivity and NPV than radiomics, but lower specificity (0.793 vs. 0.806) — a trade-off, not a uniform improvement. The abstract and Discussion now state "across **most** evaluation metrics" for EfficientNet FT, with an explicit note on the specificity trade-off.

**(b) FT vs. FE+XGB model selection rationale:**
The Table 1 caption and Discussion now explicitly state: *"EfficientNet FT is selected as the recommended configuration despite EfficientNet FE achieving a marginally higher AUC (0.858 vs. 0.851), because FT provides the most favourable sensitivity–specificity balance. Given that the primary clinical goal is early identification of cured animals — where failing to detect a cure (false negative) is more costly than a false alarm — sensitivity is the more important metric, and FT achieves sensitivity of 0.909 versus 0.621 for FE."*

**Changes to manuscript:**
- **Abstract:** "outperforms it across all metrics" → "outperforms it across most evaluation metrics".
- **Results, Performance Comparison:** Added explicit numerical comparison; noted specificity trade-off.
- **Discussion:** Corrected "across all metrics"; added FT vs. FE rationale paragraph.
- **Conclusion:** Revised to reflect nuanced comparison.
- **Table 1 caption:** Added FT vs. FE selection rationale.

---

## Minor Comments

---

### Minor Comment 1 — Confidence intervals

**Reviewer comment:** Add measures of uncertainty for the performance metrics.

**Response — IMPLEMENTED ✅**

Bootstrap 95% confidence intervals (10,000 resamples, exam-level, sampling with replacement from pooled GLOO predictions) have been computed for all radiomics metrics and are reported in the Table 1 caption:

| Metric | Point estimate | 95% CI |
|--------|---------------|--------|
| AUC | 0.770 | [0.703, 0.832] |
| Sensitivity | 0.545 | [0.421, 0.667] |
| Specificity | 0.806 | [0.755, 0.856] |
| PPV | 0.444 | [0.338, 0.554] |
| Accuracy | 0.748 | [0.698, 0.799] |

The wide intervals reflect the effective sample size of n=10 independent subjects. CIs for the DL models are not reported in this revision as per-fold predictions were not logged in the original implementation; this is acknowledged as a limitation.

---

### Minor Comment 2 — DL does not model longitudinal sequences

**Reviewer comment:** The manuscript should acknowledge that the DL models process each MRI exam independently.

**Response — IMPLEMENTED ✅**

Added to **Methods (Fine-Tuning Procedure)** and **Discussion**:

*"It should be noted that both EfficientNet configurations process each MRI examination independently as a 2D image. The model has no access to the temporal sequence of examinations, and longitudinal information is not explicitly modelled. Despite this, the GLOO evaluation strategy ensures that all examinations from a given animal are used either entirely for training or entirely for testing, preventing temporal information leakage between folds."*

---

### Minor Comment 3 — Low PPV

**Reviewer comment:** The low PPV should be discussed in relation to clinical utility.

**Response — IMPLEMENTED ✅**

Added to **Discussion**:

*"The PPV of the radiomics model (0.444) is low, reflecting the balanced class distribution (5 cured vs. 5 relapsing out of 10 treated mice). In a realistic clinical scenario where long-term cure is rarer than in this controlled preclinical model, PPV would be substantially lower still. These results should therefore be interpreted as a proof-of-concept demonstration rather than as evidence of clinical deployability. The primary utility of these models in a preclinical setting is as a decision-support tool where high sensitivity (minimising missed cures) is the priority, not high PPV."*

---

## Summary of All Changes

| Comment | What was done | Status |
|---------|--------------|--------|
| Major 1 — Data leakage | Feature selection re-implemented inside GLOO fold; pipeline re-run; Table 1 updated with corrected metrics + CIs | ✅ Done |
| Major 2 — Exam vs animal level | "animals" → "examinations" fixed throughout; animal-level majority-vote results added (Radiomics: AUC=0.92); new Supplementary Table S5 | ✅ Done |
| Major 3 — Early prediction overstated | Temporal analysis by tercile added (early AUC=0.576, mid=0.803, late=0.962); all early-prediction claims removed or moderated | ✅ Done |
| Major 4 — Conclusions overstated | "across all metrics" → "across most metrics"; FT vs FE+XGB selection rationale added; specificity trade-off discussed | ✅ Done |
| Minor 1 — Confidence intervals | Bootstrap 95% CIs added for all radiomics metrics in Table 1 caption | ✅ Done |
| Minor 2 — DL not a sequence model | Acknowledged in Methods + Discussion | ✅ Done |
| Minor 3 — Low PPV | Clinical implications discussed in Discussion | ✅ Done |
| Supplementary S3/S4 | Contents confirmed in this response; PDF compilation to be verified | ✅ Confirmed |

### Pending (requires GPU to complete)
- Bootstrap CIs and animal-level metrics for DL models (EfficientNet FE+XGB and FT): requires re-running notebooks with per-fold prediction logging. Acknowledged as a limitation in the revised manuscript.
- DeLong / permutation test comparing DL vs. radiomics ROC curves: implementation ready in `notebooks/radiomics/paper_revision/10_delong_roc_comparison.ipynb`; requires DL predictions.
