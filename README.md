Overview
This repository contains datasets and custom R scripts used to develop, validate, and evaluate a machine learning-based Clinical Decision Support System (CDSS), focusing on:
Comprehensive evaluation of 13 machine learning algorithms, with the Gradient Boosting Machine (GBM) selected as the optimal model based on AUC, calibration, and Decision Curve Analysis (DCA)
SHAP-based individualized interpretability
Incremental clinical value analysis vs. MELD/Child-Pugh scores
Clinical outcome association and subgroup analyses

Data Files
original data.csv / validation data.csv
Raw baseline clinical datasets for development and independent validation cohorts.
data_complete.csv/test.csv
Complete dataset after missing value imputation.
data_complete new.csv
Refined dataset excluding variables with severe multicollinearity (VIF > 10).
data_complete(type).csv
Dataset explicitly including specific subtypes of treatment failure (biochemical vs. clinical events).

Scripts
Development Cohort.R
Pipeline for data imputation, variable selection, comprehensive training and evaluation of 13 machine learning algorithms, and SHAP analysis for the optimal GBM model.
Validation Cohort.R
Independent validation of the final GBM model and internal subgroup analysis.
Subgroup Analysis.R
Stratified performance evaluation across various clinical subgroups.
Predictor Outcome Association.R
Statistical exploration of predictor associations with treatment failure subtypes.
Collinearity Diagnostics.R
VIF calculation and multicollinearity diagnosis for baseline candidates.
Clinical incremental value.R
Incremental prognostic value analysis (Continuous NRI, Bootstrap IDI) against baseline clinical scores.
Calibration.R
Quantitative evaluation of model calibration

Requirements
R (>= 4.2)
caret
gbm
pROC
nricens
ggplot2

Notes
Scripts and de-identified datasets are provided for reproducibility of the analysis presented in the study.
Ensure all required R packages are installed before running the pipeline.

Author
Meteorain-lucky
