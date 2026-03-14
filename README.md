# UHECR-classifier-comparison
Reproducing and extending Herrera et al. (2020) — ML-based primary particle classification of Ultra-High Energy Cosmic Rays (UHECRs) using nested cross-validation. Includes feature ablation study and ROC curve analysis across XGBoost, Random Forest, SVM, KNN, and DNN classifiers.
Key contributions:

Full reproduction of nested CV results across all five classifiers
Feature ablation study revealing Zenith angle as a stronger predictor than total particle count, contradicting the original paper's feature ranking
Improved DNN implementation achieving 0.956 accuracy
ROC curve analysis for all classifiers
