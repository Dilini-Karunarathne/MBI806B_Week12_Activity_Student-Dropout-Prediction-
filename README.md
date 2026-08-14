# Week 12 - Activity 1: Predict Students' Dropout and Academic Success

**Course:** MBI806B
**Author:** Dilini Karunarathne
**Dataset:** [UCI ML Repository #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Overview
This activity develops a Python pipeline to clean data and build a machine learning model that predicts a student's academic outcome — **Dropout**, **Enrolled**, or **Graduate** — using demographic, socio-economic, and academic performance features.

## Files in this repository
| File | Description |
|---|---|
| `Week12_Activity_Code.ipynb` | Full Python notebook: data cleaning, EDA, model training, evaluation |
| `target_distribution.png` | Class balance of the target variable |
| `correlation_heatmap.png` | Correlation between numeric features |
| `confusion_matrix.png` | Confusion matrix for the best-performing model |
| `feature_importance.png` | Top 15 most important predictive features |

## Dataset Summary
- **4,424** student records, **37** columns
- **0** missing values, **0** duplicate rows (dataset was clean, minimal preprocessing needed)
- Target distribution: **Graduate (2,209)**, **Dropout (1,421)**, **Enrolled (794)**

## Methodology
1. **Data Cleaning:** Standardised column names, checked for missing values and duplicates.
2. **EDA:** Visualised target class distribution and feature correlations.
3. **Preprocessing:** Label-encoded the target variable, applied a stratified 80/20 train-test split, and scaled features for Logistic Regression.
4. **Modelling:** Trained and compared three classifiers — Logistic Regression, Random Forest, and XGBoost.
5. **Evaluation:** Compared accuracy and weighted F1-score; generated a confusion matrix and feature importance chart for the best model.

## Results

| Model | Accuracy | Weighted F1-Score |
|---|---|---|
| Logistic Regression | 76.8% | 0.753 |
| Random Forest | 76.7% | 0.752 |
| **XGBoost (Best)** | **76.6%** | **0.762** |

## Key Insights
1. **Academic performance is the strongest predictor of dropout risk.** The number of *2nd semester curricular units approved* was the single most important feature (19.3% importance), followed by *tuition fees up-to-date status* (12.6%).
2. **Financial standing matters almost as much as grades.** Features like *scholarship holder* status and *debtor* status ranked in the top 10 predictors, indicating that a student's financial stability has a measurable effect on their likelihood of dropping out — not just their coursework performance.
3. **"Enrolled" students are the hardest to classify** (F1-score of 0.41–0.50 across models), since these students haven't yet resolved into a clear Dropout or Graduate outcome — their feature patterns overlap with both other classes.
4. **1st semester performance carries forward.** Units enrolled/approved in the 1st semester remained predictive of the final outcome even after accounting for 2nd semester performance, suggesting early intervention (after semester 1) could meaningfully reduce dropout risk.
5. **XGBoost slightly outperformed the other models** on weighted F1-score, better balancing performance across all three classes, especially the minority "Enrolled" class.

## Tools Used
Python, pandas, NumPy, scikit-learn, XGBoost, seaborn, matplotlib
