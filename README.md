# Customer Churn Prediction — XGBoost vs. MLP

A machine learning pipeline that predicts customer churn for a telecom company, comparing a gradient-boosted tree model (XGBoost) against a neural network (Multilayer Perceptron), including full hyperparameter optimization for both.

Built as part of the CSC3173 Artificial Intelligence coursework.

## Overview

Customer churn — when a customer stops using a company's service — is expensive to a business, so predicting it in advance is a classic, high-value ML use case. This project builds and compares two different modeling approaches on the same dataset to see which handles this tabular, moderately imbalanced classification problem better.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7,043 customer records with 21 features covering demographics, account details, and subscribed services. Target: whether the customer churned (`Yes`/`No`), imbalanced at roughly 73.5% / 26.5%.

## Pipeline

1. **Cleaning** — fixed a mis-typed numeric column, dropped the ID column, encoded the target
2. **Preprocessing** — one-hot encoded categorical features, stratified 80/20 train/test split, feature scaling with `StandardScaler`
3. **Modeling**
   - **XGBoost** — tuned with `GridSearchCV` (5-fold cross-validation, optimizing for F1-score)
   - **MLP (Keras)** — a feed-forward neural network, tuned via a manual search over hidden-layer sizes and learning rate
4. **Evaluation** — accuracy, precision, recall, and F1-score, both models scored on the same held-out test set

## Results

| Metric    | XGBoost | MLP    |
|-----------|---------|--------|
| Accuracy  | 0.8034  | 0.7793 |
| Precision | 0.6633  | 0.5940 |
| Recall    | 0.5267  | 0.5321 |
| F1-score  | 0.5872  | 0.5614 |

**Best XGBoost hyperparameters:** `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`
**Best MLP configuration:** hidden layers `(32, 16)`, `learning_rate=0.001`

XGBoost outperformed the MLP overall, most notably on precision — likely because tree-based ensembles tend to handle mixed categorical/numeric tabular data well without extensive tuning, while neural networks generally need more data and a larger search to match them on tasks like this.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · Keras / TensorFlow · matplotlib · seaborn · Google Colab

## What I Learned

- Building a full ML pipeline end-to-end: cleaning → preprocessing → training → hyperparameter tuning → evaluation
- Why train/test splitting and fitting scalers only on training data matters (avoiding data leakage)
- Choosing the right evaluation metric for an imbalanced classification problem
- Comparing a tree-based ensemble model against a neural network on the same tabular task

## Repository Contents

- `churn_prediction_XGBoost_MLP.ipynb` — full notebook with code, explanations, and results
