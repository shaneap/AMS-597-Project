# AMS 597 — Long Island Traffic Accident Analysis

Group project for AMS 597 (Statistical Machine Learning) at Stony Brook University.

**Team:** Shane Patandin, Matthew Tranquada, Vraj Patel, Kaushal Patel, Sur Vaghasiya

## Dataset

US Accidents (March 2023) — filtered to Long Island (Nassau & Suffolk counties).  
Cleaned dataset: `data/li_accidents_clean.csv` (33,228 records).

## Research Questions

| # | Question | Approach | Files |
|---|----------|----------|-------|
| 1 | Regression analysis of accident severity/duration | Elastic net (R) | `Regression Analysis.Rmd` |
| 2 | Spatial clustering of accident hotspots on Long Island | K-means / hierarchical with SHAP explanation (R) | `Longisland_Clustering/` |
| 3 | Binary severity prediction (high vs. low) | Logistic Regression, XGBoost | `Severity_Prediction/`

## Repository Structure

```
├── data/
│   ├── US_Accidents_March23.csv       Raw national file (gitignored, ~3 GB)
│   └── li_accidents_clean.csv         Cleaned Long Island subset
│
├── cleaning/                          Data cleaning pipeline
│   ├── cleaning_dataset.Rmd           Reproducible cleaning workflow
│   ├── cleaning_walkthrough_concise.pdf
│   └── cleaning_walkthrough_detailed.pdf
│
├── Longisland_Clustering/             Research Question 2 — Spatial Clustering
│   ├── li_clustering_v3.Rmd           Current analysis (SHAP cluster explanation)
│   ├── planning/                      SRS, execution plans, audit docs
│   └── archive/                       Superseded v1/v2 files
│
├── Severity_Prediction/               Research Question 3 — Severity Prediction
│   ├── Severity_Prediction.ipynb      Logistic Regression + XGBoost (Python)
│   ├── Severity_Prediction.md         Analysis write-up
│   ├── Severity_Prediction.html       Rendered HTML report
│   └── Severity_Prediction.pdf        Rendered PDF report
│
├── figures/                           Output figures
│   ├── confusion_matrices.png
│   ├── cv_comparison.png
│   ├── pr_curve.png
│   ├── roc_curves.png
│   └── xgb_feature_importance.png
│
├── docs/                              Admin & submission documents
│   ├── AMS_597_Group_Dataset_Submission.pdf
│   └── AMS_597_Group_Members.pdf
│
├── Regression Analysis.Rmd            Research Question 1 — Regression (R)
└── Group Project.Rproj                RStudio project file
```

## Links

- [Google Doc](https://docs.google.com/document/d/1NEaOt-gkrq0qIp9cybiw-GyIrrtHDJKECGdPcDdgxr8/edit?usp=sharing)
- [Google Drive](https://drive.google.com/drive/folders/1pS0WjtN5MeBijuFsyNB3RsR_g0xBnTVn?usp=sharing)
