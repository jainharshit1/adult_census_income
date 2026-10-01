# CS 660 Final Project: Adult Census Income Prediction

This repository contains an end-to-end machine learning pipeline developed for the CS 660 (Data Mining for Decision Making) final project. The objective is to predict whether an individual's annual income exceeds $50K using demographic and employment data from the UCI Adult Census dataset.

## Dataset Overview
* **Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
* **Task:** Binary Classification (`<=50K` vs `>50K`)
* **Instances:** 48,842 total observations

## Data Preprocessing Pipeline
To prepare the data for distance-based and tree-based algorithms, the following engineering and cleaning transformations are applied:
* **Outlier Transformation:** Applied a `log1p` transformation to `capital-gain` and `capital-loss` to mathematically compress extreme artificial spikes (e.g., the $99,999 cap) and normalize the distribution.
* **Cardinality Reduction:** Grouped 42 highly sparse `native-country` values into 5 dense geographic regions (North America, Latin America, Europe, Asia, Other) to eliminate feature sparsity.
* **Feature Pruning:** Dropped the `fnlwgt` (census sampling weight) and `education` (redundant to `education-num`) columns to prevent multicollinearity and noise.
* **Imputation:** Addressed missing values (originally encoded as `?`) using a `SimpleImputer` with a `most_frequent` strategy.
* **Class Balancing:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) strictly on the training set to resolve the 76/24 class imbalance, generating a perfectly balanced 50/50 split for robust model training.

## Modeling and Sensitivity Analysis
The project evaluates 10 distinct machine learning algorithms. A core focus of this study is rigorous hyperparameter tuning (Sensitivity Analysis) to maximize classification accuracy and observe the bias-variance tradeoff.

Key models highlighted in the presentation include:
* **Support Vector Machines (SVM):** Analyzed the geometric impact of adjusting regularization (`C`) and the radial basis function's radius of influence (`gamma`) on the decision boundary.
* **Gradient Boosting Classifier:** Evaluated the sequential error-correction tradeoff by visualizing `learning_rate` against `n_estimators`.
* **Logistic Regression & Naive Bayes:** Tuned regularization strength and smoothing parameters to establish calibrated, interpretable baselines.

All models are tuned using `GridSearchCV` with 5-fold cross-validation, evaluated on a strictly isolated test set, and scored using Accuracy, ROC-AUC, and F1-Scores against published UCI baselines.

## Requirements and Installation
The pipeline is built in Python. To install the required dependencies, run the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
