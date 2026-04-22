---
output:
  pdf_document: default
  html_document: default
---
# Traffic Safety Analysis on Long Island
**AMS 597 — Statistical Computing | Spring 2026**  
Shane Patandin · Matthew Tranquada · Vraj Patel · Kaushal Patel · Sur Vaghasiya  
Stony Brook University

---

## 1. Introduction

This report analyzes traffic accident patterns on Long Island using the US Accidents dataset (Moosavi et al., 2023), filtered to Nassau, Suffolk, and Queens counties. The final working dataset contains 33,228 accidents across 48 variables spanning March 2016 to March 2023. Variables cover four domains: temporal (time of day, day of week, season), weather (temperature, humidity, visibility, precipitation type), infrastructure (junction presence, traffic signals, crossings), and outcome (severity on a 1–4 scale). The dataset is imbalanced — roughly 85% of records are Severity 2.

Three research questions are addressed:

1. **[Research Question 1 — to be completed]**
2. Can unsupervised clustering reveal distinct accident archetypes on Long Island, and do those archetypes differ in severity risk?
3. Can machine learning models predict whether a crash will be high-severity (Severity 3–4) using only information available at the time of the accident?

Research Questions 1 and 2 are implemented in R (RMarkdown); Research Question 3 uses Python (Jupyter Notebook).

### 1.1 Data Preprocessing

Raw data required several cleaning steps before modeling:

- **Missing values:** Numeric weather variables were median-imputed. Precipitation was excluded from clustering models due to 20.1% missingness.
- **Outliers:** Accident duration was 99th-percentile capped and log-transformed.
- **Feature engineering:** Temporal features (hour_band, rush_hour, season, Sunrise_Sunset) and 7-level weather groupings (collapsed from 37 raw labels) were constructed. Cyclical sine/cosine encodings were applied to hour-of-day and month to avoid wrap-around discontinuities. Interaction terms (night×rain, rush_hour×junction) were added to the predictive models.
- **Lookahead exclusions:** Distance(mi) and Duration_Minutes were excluded from the predictive model since both are only measurable after a crash resolves.

---

> **[ Section 1 — Research Question 1 placeholder ]**
> *Insert the full RQ1 write-up here. Estimated length: 6–8 pages. The combined report should remain under 25 pages excluding appendices.*

---

## 2. Research Question 2: Unsupervised Accident Clustering

**Question:** Can unsupervised learning reveal distinct accident archetypes on Long Island based on weather and road characteristics, and do those archetypes differ in severity risk?

This is approached through two parallel tracks — scenario clustering (what type of accident?) and spatial clustering (where does it concentrate?) — then cross-tabulated to identify which accident profiles dominate each highway corridor. Severity is excluded from all clustering inputs and used only afterward for post-cluster profiling.

### 2.1 Methods

#### Scenario Clustering: k-Prototypes (Baseline)

k-Prototypes extends k-Means to mixed data by combining Euclidean distance for continuous variables with simple matching distance for categorical ones. Candidates at k $\in$ {3,...,8} were fit with Gower range-normalisation, nstart=10, and seed 42.

The baseline produced a silhouette of 0.1616 at k = 3 — well below the 0.30 acceptance threshold (Kaufman & Rousseeuw, 1990) — and a Jaccard bootstrap stability of 0.243, far below the 0.60 acceptable level (Hennig, 2007). Figure 1 shows the silhouette curve across all candidate k values, none of which clear the threshold.

![Figure 1: k-Prototypes silhouette validation across candidate k values.](figures/c_kproto_silhouette.png)

*Figure 1: k-Prototypes silhouette validation. No configuration clears the 0.30 acceptance threshold, classifying all findings as exploratory.*

#### Scenario Clustering: FAMD + k-Means

To improve separation, a two-stage dimensionality reduction strategy guided by SHAP feature importance was implemented.

A preliminary 500-tree Random Forest ranked all 17 scenario features by permutation importance. The bottom 9 features — weekday, log_Duration, weekend, Wind_Speed, Pressure, Crossing, Traffic_Signal, Junction, Stop — all had near-zero scores and were dropped. The remaining 8 features (rush_hour, hour_band, Sunrise_Sunset, Visibility(mi), weather_group, Humidity(%), Temperature(F), season) were passed into Factor Analysis of Mixed Data (FactoMineR::FAMD), which projects mixed-type features into orthogonal numeric components via indicator expansion.

A grid search over 20 (ncomp × k) combinations evaluated silhouette width at each setting. Four components with k = 4 produced the best silhouette (0.4268); adding more components degraded separation monotonically. Figure 2 shows the silhouette curve for the FAMD pipeline — all candidates clear the threshold, with k = 4 as the clear optimum.

![Figure 2: FAMD + k-Means silhouette width by k.](figures/c_silhouette_by_k.png)

*Figure 2: FAMD + k-Means silhouette width across candidate k values. All configurations exceed the 0.30 threshold; k = 4 achieves the maximum at 0.4268.*

Final clustering used k-Means on the 4-dimensional FAMD coordinate matrix (nstart=25, iter.max=100). Figure 3 shows the biplot of the two leading FAMD dimensions coloured by cluster assignment — the four clusters separate clearly in this reduced space.

![Figure 3: FAMD biplot coloured by cluster assignment (k = 4).](figures/c_famd_biplot.png)

*Figure 3: FAMD biplot (Dim 1 vs. Dim 2) coloured by cluster. The 68% concentration ellipses confirm meaningful geometric separation between all four archetypes.*

#### Spatial Clustering: DBSCAN

DBSCAN was applied to km-projected coordinates anchored at (40.8°N, 73.2°W). A 68-combination grid search over eps $\in$ [0.3, 2.0] km and minPts $\in$ {40, 50, 75, 100} selected eps = 1.0 km and minPts = 100.

### 2.2 Results

#### Method Comparison

| Method | k | Silhouette | Stability |
|---|---|---|---|
| k-Prototypes (Gower) | 3 | 0.1616 | 0.243 (Jaccard) |
| **FAMD + k-Means** | **4** | **0.4268** | **0.997 (ARI)** |

The FAMD pipeline improves silhouette by 165% and achieves near-perfect bootstrap reproducibility (ARI = 0.997 over 20 resamples), upgrading the quality gate from exploratory to acceptable. The improvement comes from three reinforcing factors: feature selection removes noise that Gower distance weights equally to informative features; FAMD concentrates discriminative variation into orthogonal dimensions; and k-Means in Euclidean space has stronger convergence guarantees than k-Prototypes.

#### The Four Accident Archetypes

| C | Name | Size | Sev4% | Night% | Wknd% | Dominant Conditions | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Cloudy Off-Peak | 7,220 | 2.9 | 0 | 19 | Cloudy, Midday, 64°F, vis 9.9 mi | Baseline |
| 2 | AM Rush | 16,754 | 1.8 | 13 | 10 | Cloudy, AM Rush, Junction-heavy | Low |
| 3 | Rain AM Rush | 3,730 | 2.1 | 23 | 12 | Rain (51%), vis 3.1 mi | Baseline |
| **4** | **Night / Early AM** | **5,524** | **6.1** | **97** | **29** | Cloudy, 53°F, vis 9.5 mi | **High** |

Figure 4 presents the cluster fingerprint heatmap — a standardised view of each archetype's feature profile. Cluster 4's night share (97%) and elevated Severity 4 rate (6.1%) stand out immediately against the other three.

![Figure 4: Scenario cluster fingerprint heatmap.](figures/c_fingerprint_heatmap.png)

*Figure 4: Standardised cluster fingerprints. Tile colour is scaled within each row. C4's night share (97%) and Severity 4 rate (6.1%) are the dominant signals.*

Cluster 4 is the primary risk signal: a 6.1% Severity 4 rate is 2.21× the dataset baseline of 2.76%. Nearly all accidents (97%) occur at night, and the 29% weekend share is the highest of any cluster, suggesting recreational late-night driving contributes disproportionately to severe outcomes.

#### SHAP Feature Importance

Figure 5 shows global permutation importance from the Random Forest trained to predict cluster membership (OOB accuracy = 99.6%).

![Figure 5: Global SHAP feature importance for cluster separation.](figures/c_shap_global.png)

*Figure 5: Global permutation importance (500-tree Random Forest, OOB error = 0.005). rush_hour alone accounts for one-third of classification power.*

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | rush_hour | 0.330 | Single strongest separator |
| 2 | hour_band | 0.151 | Finer time-of-day resolution |
| 3 | Sunrise_Sunset | 0.104 | Day vs. night — drives C4 |
| 4 | Visibility(mi) | 0.098 | Separates rain (C3) from clear (C1) |
| 5 | weather_group | 0.045 | Rain vs. Cloudy vs. Clear |
| 6 | Humidity(%) | 0.021 | Moisture conditions |
| 7 | Temperature(F) | 0.011 | Seasonal temperature signal |
| 8 | season | 0.004 | Winter vs. summer baseline |

Figure 6 breaks this down per cluster, showing that each archetype is driven by a distinct feature profile — confirming the four groups are genuinely different in character rather than arbitrary partitions.

![Figure 6: Per-cluster SHAP feature importance (one-vs-rest).](figures/c_shap_per_cluster.png)

*Figure 6: Per-cluster permutation importance. C2 is defined almost entirely by rush_hour; C3 by visibility and weather_group; C4 by Sunrise_Sunset alongside temporal features.*

rush_hour alone accounts for one-third of the Random Forest's classification power. The top three features are all temporal — *when* you drive matters substantially more than what the weather is.

#### Spatial Hotspots (DBSCAN)

Twelve corridor clusters were identified along Long Island's major east–west highways. Figure 7 shows the spatial distribution overlaid on Nassau and Suffolk county boundaries.

![Figure 7: DBSCAN spatial hotspot clusters on Long Island.](figures/c_dbscan_map.png)

*Figure 7: DBSCAN spatial clusters (eps = 1.0 km, minPts = 100). Twelve corridor clusters trace Long Island's major highways; 15.3% of points are noise (grey).*

The three largest corridors are the Northern State Parkway (44.3% of corridor-assigned accidents), Southern State Parkway (33.2%), and I-495/LIE (3.3%). Cross-tabulation: the AM Rush archetype (C2) dominates 11 of 12 hotspot corridors, confirming that most corridor-level congestion is daytime commuter-driven rather than weather or nighttime events.

#### Validation

| Metric | Score | Threshold | Verdict |
|---|---|---|---|
| Silhouette width | 0.4268 | $\geq$ 0.30 | PASS |
| Bootstrap ARI (20 reps) | 0.997 | $\geq$ 0.65 | PASS |
| ARI std. dev. | 0.0013 | — | Near-perfect |
| RF classification accuracy | 99.6% | — | PASS |
| DBSCAN corridor count | 12 | 3–15 | PASS |
| DBSCAN noise fraction | 15.3% | 5–30% | PASS |

### 2.3 Limitations

- FAMD retains 39.8% of total variance. Additional components were tested and degrade silhouette, so this is a deliberate trade-off rather than information loss.
- All results describe associations, not causal relationships.
- Precipitation was excluded due to 20.1% missingness, removing a potentially important weather signal.
- With Severity 2 comprising 85% of records, subtle differences among severe accident types may be masked.
- DBSCAN corridor labels are heuristic and were not validated against authoritative highway shapefiles.

---

## 3. Research Question 3: Accident Severity Prediction

**Question:** Can machine learning models trained on environmental and infrastructure features predict whether a Long Island crash will be high-severity (Severity 3–4), using only information available at the time of the accident?

Two models are compared: Logistic Regression as an interpretable baseline and XGBoost as a gradient-boosted ensemble. Given the 5.86:1 class imbalance, ROC-AUC is the primary metric. F1-score on the high-severity class and the Precision-Recall curve are reported as supplementary diagnostics.

### 3.1 Data and Features

The dataset covers Nassau, Suffolk, and Queens counties. The outcome variable severity_binary bins the four-level scale into low (1–2, coded 0) and high (3–4, coded 1). The test set contains 6,646 records with 969 high-severity cases (class ratio $\approx$ 5.86:1).

Twenty-six features were constructed across four domains:

| Domain | Features | Notes |
|---|---|---|
| Weather (continuous) | Temperature, Humidity, Pressure, Visibility, Wind Speed | NAs imputed post-split |
| Temporal | hour_sin/cos, month_sin/cos, dayofweek_sin/cos, rush_hour, is_weekend, is_night | Cyclical encoding prevents hour-wrap discontinuity |
| Infrastructure | Junction, Traffic_Signal, Crossing, Stop | Boolean flags re-encoded as 0/1 |
| Weather (categorical) | wx_Cloudy, wx_Rain, wx_Snow_Ice, wx_Fog_Haze, wx_Thunderstorm, wx_Other | One-hot; Clear dropped as reference |
| Interactions | night×rain, night×fog, rush_hour×junction | Compound risk factors |

### 3.2 Modelling Approach

Data were split 80/20 with stratification on the outcome. All preprocessing (scaling, imputation) is encapsulated in scikit-learn Pipelines fitted only on training data.

**Logistic Regression:** StandardScaler → KNNImputer (k=5) → L1-penalised LR (liblinear, max_iter=500, class_weight='balanced'). Regularisation strength C tuned over {0.01, 0.1, 1.0, 10.0, 100.0} via 5-fold stratified CV on ROC-AUC.

**XGBoost:** Same preprocessing plus a SelectFromModel step (threshold = mean gain) using a preliminary 100-tree classifier to prune features. Final model uses 300 estimators, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.86. max_depth $\in$ {3, 5, 7} and learning_rate $\in$ {0.01, 0.05, 0.1} jointly tuned via the same CV procedure.

### 3.3 Results

#### Hold-Out Performance

| Model | ROC-AUC | Accuracy | F1 (High Sev.) |
|---|---|---|---|
| Logistic Regression | 0.6165 | 0.5846 | 0.2955 |
| **XGBoost** | **0.7185** | **0.7040** | **0.3706** |

Figure 8 shows the ROC curves on the held-out test set. XGBoost dominates across the full threshold range; both models sit well above the random-classifier diagonal.

![Figure 8: ROC curves — XGBoost vs. Logistic Regression.](figures/x_roc_curves.png)

*Figure 8: ROC curves on the held-out test set. XGBoost AUC = 0.718; Logistic Regression AUC = 0.616.*

XGBoost outperforms across all three metrics. The 0.102 AUC gap points to non-linear structure that a linear model cannot fully represent even with engineered interaction terms. Accuracy is not the headline metric: a model that always predicted low-severity would hit ~85% accuracy while achieving AUC = 0.50.

Figure 9 shows the confusion matrices. Both models flag the same 579 true high-severity crashes, but XGBoost cuts false positives from 2,371 to 1,577 — a 34% reduction that accounts for most of the accuracy and F1 improvement.

![Figure 9: Confusion matrices on the held-out test set.](figures/x_confusion_matrices.png)

*Figure 9: Confusion matrices (n = 6,646). Both models catch 579 true high-severity crashes; XGBoost generates 794 fewer false positives.*

#### Precision-Recall Analysis

Figure 10 shows the PR curves. Unlike ROC, the PR curve ignores true negatives entirely, making it a more conservative read of performance under class imbalance. Both models clear the random baseline (AP $\approx$ 0.15); XGBoost's advantage is most pronounced in the 0.0–0.4 recall range.

![Figure 10: Precision-Recall curves.](figures/x_pr_curve.png)

*Figure 10: Precision-Recall curves. XGBoost AP = 0.327; Logistic Regression AP = 0.213; random baseline AP $\approx$ 0.15.*

#### Feature Importance

Figure 11 shows the top-15 XGBoost features by mean gain. Wind Speed leads at 0.085, followed closely by Crossing and Junction. Both engineered interaction terms land in the top ten, confirming they carry signal beyond their constituent features.

![Figure 11: XGBoost feature importance by mean gain (top 15).](figures/x_feature_importance.png)

*Figure 11: Top-15 XGBoost features by mean gain. Infrastructure features (Crossing, Junction, Traffic_Signal) and temporal encodings dominate alongside Wind Speed.*

#### Cross-Validation Stability

Figure 12 shows 5-fold CV AUC with error bars. CV results track test performance closely — LR at 0.6129 (±0.0059) vs. test 0.6165, XGBoost at 0.7164 (±0.0042) vs. test 0.7185 — confirming stable, generalisable models.

![Figure 12: 5-fold cross-validation AUC comparison.](figures/x_cv_auc.png)

*Figure 12: 5-fold CV ROC-AUC with ±1 SD error bars. Tight standard deviations and close agreement with test-set results indicate neither model is overfitting.*

### 3.4 Limitations

- The data is observational. Wind Speed's top rank likely reflects correlation with broader adverse conditions not fully captured in the feature set.
- Analysis is limited to Nassau, Suffolk, and Queens. Feature importances may not transfer to other regions.
- Class imbalance was handled through loss re-weighting. Resampling methods like SMOTE weren't evaluated.
- There is no temporal holdout. A time-based split would give a cleaner picture of prospective performance.
- The default 0.5 threshold is used throughout. Any real deployment would need an explicit operating point from the PR curve.

---

## 4. Real-World Applications

Across both analyses, the inputs are all observable before a crash occurs, which means the framework naturally points toward prevention. Three application areas follow from the results.

### 4.1 Infrastructure Investment

Crossing, Junction, and Traffic_Signal rank among the top seven XGBoost features by gain, and the rush_x_junction interaction ranks ninth. Certain infrastructure types — particularly marked crossings and multi-leg junctions — are disproportionately associated with high-severity outcomes, and the effect amplifies during peak-hour congestion.

This gives Nassau County DOT, Suffolk County DPW, and NYC DOT (Queens) a way to prioritise safety capital budgets beyond simple crash-count rankings. Weighting candidate sites by predicted severity risk produces a different and more consequential ordering. Practical interventions at flagged locations could include signal timing adjustments during identified peak-risk windows, advanced warning signage approaching high-gain junctions, turn lane additions and pedestrian refuge islands at high-risk crossings, and retroreflective markings calibrated to low-visibility conditions.

The DBSCAN spatial results (Figure 7) sharpen the geographic targeting. The AM Rush archetype (C2) dominates 11 of 12 corridor clusters and accounts for 50% of all accidents by volume. Even at a low individual severity rate (1.8% Severity 4), the volume makes this the costliest cluster in aggregate. Congestion management and junction redesign at the densest Northern/Southern State Parkway segments would reduce the largest source of daily traffic disruption on Long Island.

### 4.2 Safety Campaigns

The clustering analysis identifies Cluster 4 (Night / Early AM) as the clearest public safety target: 17% of all accidents but a 6.1% Severity 4 rate — 2.21× the baseline. The 29% weekend share suggests recreational late-night driving contributes disproportionately to the most dangerous outcomes (see Figure 4).

Key campaign directions:

- **Nighttime messaging** focused on the 10 PM–5 AM window along the LIE and Northern/Southern State corridors, with Friday and Saturday nights as the highest-yield intervention windows.
- **Impaired-driving enforcement:** The late-night weekend profile is consistent with alcohol- and fatigue-related risk. DUI checkpoints during these hours are directly supported by the data.
- **Road lighting improvements:** C4 occurs overwhelmingly under clear conditions (mean visibility 9.5 mi), ruling out weather as the primary driver. Reduced ambient light is the differentiating factor.
- **Wet-weather campaigns:** The Rain AM Rush archetype (C3) involves 51% rain prevalence and reduced visibility. Autumn and winter campaigns reminding drivers to reduce speed in wet conditions address this cluster's profile.

### 4.3 Real-Time Driver Alerts

All XGBoost inputs — wind speed, hour, month, infrastructure type, weather conditions — are available in real time from weather APIs and map data. A (location, timestamp, weather observation) tuple could be scored against the trained model and, if the predicted high-severity probability exceeds a chosen threshold, a targeted advisory issued to navigation apps or dynamic highway signs.

This is more specific than a generic weather advisory because it accounts for infrastructure context. A driver approaching a signalised crossing during evening rush under high-wind conditions lands well above average predicted severity — a trigger that a "fog advisory" would miss entirely if fog is absent.

At the default 0.5 threshold, XGBoost achieves 0.60 recall with 0.27 precision — roughly one in four alerts corresponds to a genuinely high-severity situation. For a driver advisory system that's a reasonable operating point, and it can be shifted along the PR curve (Figure 10) depending on how conservative the deployment needs to be. The clustering model adds a complementary layer: Sunrise_Sunset and rush_hour features can trigger time-specific alert modes (e.g., heightened sensitivity after 10 PM on weekends) even before conditions deteriorate.

---

## 5. Conclusion

The clustering analysis elevated scenario clustering from exploratory (silhouette 0.16, Jaccard 0.24) to acceptable (silhouette 0.43, ARI = 0.997) by combining SHAP-guided feature selection with FAMD dimensionality reduction. Four accident archetypes were identified. The Night/Early AM cluster carries a Severity 4 rate of 6.1% — more than double the dataset baseline — and is the strongest risk signal in the data. DBSCAN confirmed 12 spatial corridor clusters, with the AM Rush archetype dominating 11 of them. The central finding is that *when* you drive is the dominant risk factor on Long Island.

The severity prediction analysis showed that XGBoost (AUC = 0.718) meaningfully outperforms Logistic Regression (AUC = 0.616), with the gap driven by non-linear interactions involving wind speed, infrastructure type, and time-of-day. Infrastructure features (Crossing, Junction, Traffic_Signal) rank consistently among the top predictors, and the rush_x_junction interaction confirms that junction risk concentrates during peak hours. Cross-validation results are stable and consistent with test performance throughout.

The three application domains — infrastructure investment prioritisation, targeted safety campaigns, and real-time driver alerts — all work with information available before a crash happens. The models give agencies and designers a principled basis for deciding where and when to intervene.

### Overall Limitations

- All analyses use observational data. Identified associations do not imply causation, and unmeasured confounders (road surface condition, driver behaviour, vehicle type) may explain part of the patterns.
- Geographic scope is limited to Long Island and Queens. Findings may not generalise to regions with different road networks.
- Temporal coverage ends at March 2023 and does not reflect post-pandemic traffic shifts or recent infrastructure changes.
- Precipitation data were excluded from clustering models due to high missingness, removing a potentially important weather signal.

---

## References

Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data Analysis, 52*(1), 258–271.

Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.* Wiley.

Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., & Ramnath, R. (2023). A countrywide traffic accident dataset. *arXiv:1906.05409.*

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*

Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R package for multivariate analysis. *Journal of Statistical Software, 25*(1), 1–18.
