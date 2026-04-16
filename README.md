# 🏦 Adult Income Classification — End-to-End ML Pipeline

A complete supervised machine learning pipeline to predict whether an individual's annual income exceeds **$50K**, using the U.S. Census Bureau's Adult Income dataset. The project covers everything from exploratory data analysis to hyperparameter-tuned model evaluation and model serialization.

---

## 📌 Problem Statement

The U.S. Census Bureau collects demographic and employment data from individuals. This project formulates a **Supervised Binary Classification** task:

- **Input**: Demographic and employment attributes (age, education, occupation, hours worked per week, capital gain/loss, marital status, etc.)
- **Target**: Whether an individual earns `>50K` (1) or `≤50K` (0) per year

**Use cases:**
- Financial institutions — loan eligibility & credit risk assessment
- Government agencies — social welfare benefit allocation
- Market researchers — high-value customer segmentation
- HR/Recruitment firms — salary benchmarking

---

## 📂 Project Structure

```
Final_Adult_Income_Classification/
│
├── app/                                # Deployment / inference application
│
├── Datasets/                           # Raw and processed data files
│   └── adult_income.csv
│
├── Models/                             # Saved model pipelines
│   └── adult_income_best_model.pkl
│
├── Notebooks/                          # Jupyter notebooks
│   └── Adult_Income_Classification.ipynb
│
├── Visualizations/                     # All EDA and evaluation plots
│   ├── 01_target_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── 03_categorical_distributions.png
│   ├── 04_bivariate_numerical.png
│   ├── 05_categorical_vs_target.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_outlier_treatment.png
│   ├── 08_mutual_information.png
│   ├── 09_metrics_comparison.png
│   ├── 10_confusion_matrices.png
│   ├── 11_roc_curves.png
│   ├── 12_pr_curves.png
│   └── 13_metrics_heatmap.png
│
└── requirements.txt                    # Python dependencies
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual |
| `workclass` | Categorical | Type of employer (Private, Govt, Self-emp, etc.) |
| `education` | Categorical | Highest education level |
| `education.num` | Numerical | Education encoded as number (1–16) |
| `marital.status` | Categorical | Marital status |
| `occupation` | Categorical | Job category |
| `sex` | Categorical | Gender |
| `capital.gain` | Numerical | Investment capital gain |
| `capital.loss` | Numerical | Investment capital loss |
| `hours.per.week` | Numerical | Weekly working hours |
| `income` | **Target** | `≤50K` (0) or `>50K` (1) |

> **Dropped features**: `fnlwgt` (weak predictor), `race` (low MI score), `native.country` (high cardinality), `relationship` (redundant with `marital.status`), `education` (redundant with `education.num`)

**Dataset size**: ~48,842 records | **Class balance**: ~75% ≤50K / ~25% >50K (moderate imbalance)

---

## 🔧 Pipeline Overview

```
Raw Data
   │
   ├── Step 1: Business Problem Definition
   ├── Step 2: Load Dataset & Import Libraries
   ├── Step 3: Descriptive Analysis & Basic Checks
   ├── Step 4: Exploratory Data Analysis (EDA)
   ├── Step 5: Data Preprocessing
   │     ├── Handle missing values (mode imputation)
   │     ├── Drop duplicates
   │     ├── Encode target variable
   │     ├── Outlier treatment (IQR clipping, 3 iterations)
   │     └── Drop redundant features
   ├── Step 6: Feature Selection (Mutual Information)
   ├── Step 7: Train-Test Split + Preprocessing Pipeline
   ├── Step 8: Model Building (10 Classifiers)
   ├── Step 9: Hyperparameter Tuning (RandomizedSearchCV)
   ├── Step 10: Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
   ├── Step 11: Best Model Detailed Analysis
   └── Step 12: Save Best Model as .pkl
```

---

## 🤖 Models Trained

| # | Model | Notes |
|---|---|---|
| 1 | **K-Nearest Neighbors** | Instance-based, no distributional assumptions |
| 2 | **Naive Bayes** | Fast probabilistic baseline |
| 3 | **Decision Tree** | Interpretable, non-linear boundaries |
| 4 | **Support Vector Machine** | Trained on subset (8K rows) for speed |
| 5 | **Logistic Regression** | Linear baseline, balanced class weights |
| 6 | **Random Forest** | Robust ensemble with built-in feature importance |
| 7 | **Gradient Boosting** | Sequential boosting on tabular data |
| 8 | **XGBoost** | Regularized boosting with `scale_pos_weight` |
| 9 | **LightGBM** | Fastest boosting, leaf-wise tree growth |
| 10 | **LightXGB Boosting** | Soft-voting ensemble of LightGBM + XGBoost |

All models use an **imbalanced-learn Pipeline** with **SMOTE** to ensure synthetic oversampling only occurs within training folds — preventing data leakage.

---

## 📈 Evaluation Metrics

Since the dataset is moderately imbalanced, accuracy alone is not sufficient. The pipeline evaluates models on:

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Baseline sanity check |
| **Precision** | Avoids false-positive targeting of >50K individuals |
| **Recall** | Captures as many true >50K individuals as possible |
| **F1-Score** | Primary metric — balances Precision & Recall |
| **AUC-ROC** | Ranking ability across all classification thresholds |

---

## 🏆 Results

| Rank | Model | F1-Score | AUC-ROC |
|---|---|---|---|
| 🥇 1st | **Gradient Boosting** | **0.6661** | **0.8878** |
| 🥈 2nd | LightGBM / XGBoost | — | — |
| 🥈 3rd | LightXGB Boosting | — | — |
| 🥉 4th | Random Forest | — | — |
| 5th | Logistic Regression | — | — |
| 6th | Decision Tree | — | — |
| 7th | KNN | — | — |
| 8th | Naive Bayes | — | — |
| 9th | SVM | — | — |

> Boosting models consistently outperformed other algorithms on this tabular census dataset.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn scipy
```

### Run the Notebook

```bash
git clone https://github.com/your-username/Final_Adult_Income_Classification.git
cd Final_Adult_Income_Classification
jupyter notebook Adult_Income_Classification.ipynb
```

---

## 💾 Load the Saved Model

```python
import pickle

with open('adult_income_best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

# Predict on new data
pipeline = bundle['pipeline']
predictions = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)[:, 1]
```

---

## 🔑 Key Takeaways

1. **SMOTE inside ImbPipeline** — Synthetic samples never leak into the test set.
2. **Top predictors** — `capital.gain`, `marital.status`, and `education.num` carried the most signal.
3. **RandomizedSearchCV + StratifiedKFold(5)** — Robust and unbiased hyperparameter tuning.
4. **F1-Score & AUC-ROC** are the correct primary metrics — a dummy classifier already achieves 75% accuracy by predicting all ≤50K.
5. **Boosting models** (XGBoost, LightGBM, Gradient Boosting) consistently outperformed all other algorithms on this tabular dataset.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset source: [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), and [imbalanced-learn](https://imbalanced-learn.org/)
