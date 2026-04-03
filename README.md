# EEG-Based Psychiatric Disorder Classification

A machine learning project that classifies psychiatric disorders from EEG (electroencephalogram) brainwave data. Built as a semester project for CS 360 (Machine Learning) at Bradley University, Fall 2025.

## Overview

This project analyzes EEG recordings from 1,142 individuals to classify 7 psychiatric disorder categories using supervised learning. The approach compares multi-class classification (all disorders simultaneously) against binary classification (individual disorder vs. healthy controls), finding that binary models significantly outperform multi-class across all conditions.

## Results

| Classification Type | Model | Accuracy |
|---|---|---|
| Multi-class (7 disorders) | Random Forest | ~31.5% |
| Binary (per disorder) | Random Forest | 68–82% |
| Multi-class (7 disorders) | KNN + PCA | ~28% |

Key findings:
- Binary classification consistently outperformed multi-class across all 7 disorders
- Coherence features were more discriminative than absolute power features for most conditions
- Schizophrenia binary classification accuracy (~68%) matched published research benchmarks
- Class imbalance handled with `class_weight='balanced'` for underrepresented disorders

## Dataset

[EEG Psychiatric Disorders Dataset](https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset) from Kaggle — 1,142 patients, resting-state eyes-closed EEG recordings with 114 spectral features.

## Tech Stack

- **Python 3** — Pandas, NumPy, Matplotlib, Seaborn
- **scikit-learn** — Random Forest, KNN, PCA, StandardScaler, cross-validation
- **Jupyter Notebook** — Interactive development and visualization

## Methods

1. **Data Cleaning** — Dropped missing values in education/IQ fields, isolated EEG features from demographic data
2. **Feature Engineering** — Used 114 EEG spectral features (coherence and absolute power bands)
3. **Multi-class Classification** — KNN with hyperparameter tuning (k=1–100) and PCA dimensionality reduction; Random Forest (250 estimators)
4. **Binary Classification** — Trained separate Random Forest models (500 estimators) for each disorder vs. healthy controls
5. **Evaluation** — 5-fold cross-validation, confusion matrices, feature importance analysis for each disorder

## Project Files

- `WorkBook.ipynb` — Main analysis notebook with all models and visualizations
- `EEG.machinelearing_data_BRMH.csv` — Source dataset
- `CS360_Project_FA25.docx` — Project writeup and analysis

## License

MIT
