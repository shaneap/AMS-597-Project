# AMS 597 — Long Island Traffic Accident Analysis

Group project for AMS 597 at Stony Brook University.

**Team:** Shane Patandin, Matthew Tranquada, Vraj Patel, Kaushal Patel, Sur Vaghasiya

## Dataset

US Accidents (March 2023) — filtered to Long Island (Nassau & Suffolk counties).  
Cleaned dataset: `data/li_accidents_clean.csv` (33,228 records).

## Repository Structure

```
├── data/                         Shared datasets
│   ├── US_Accidents_March23.csv  Raw national file (gitignored, ~3 GB)
│   └── li_accidents_clean.csv    Cleaned Long Island subset
│
├── cleaning/                     Data cleaning pipeline
│   ├── cleaning_dataset.Rmd      Reproducible cleaning workflow
│   ├── cleaning_walkthrough_concise.pdf
│   └── cleaning_walkthrough_detailed.pdf
│
├── Longisland_Clustering/        Research Question 3 — Clustering
│   ├── li_clustering_v3.Rmd      Current analysis (v3)
│   ├── output/                   Runtime artifacts (gitignored)
│   ├── planning/                 SRS, execution plans, audit docs
│   └── archive/                  Superseded v1/v2 files
│
├── docs/                         Admin & submission documents
│   ├── AMS_597_Group_Dataset_Submission.pdf
│   └── AMS_597_Group_Members.pdf
│
└── Group Project.Rproj           RStudio project file
```

## Links

- [Google Doc](https://docs.google.com/document/d/1NEaOt-gkrq0qIp9cybiw-GyIrrtHDJKECGdPcDdgxr8/edit?usp=sharing)
- [Google Drive](https://drive.google.com/drive/folders/1pS0WjtN5MeBijuFsyNB3RsR_g0xBnTVn?usp=sharing)
