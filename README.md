# UHECR-classifier-comparison
Reproducing and extending Herrera et al. (2020) — ML-based primary particle classification of Ultra-High Energy Cosmic Rays (UHECRs) using nested cross-validation. Includes feature ablation study and ROC curve analysis across XGBoost, Random Forest, SVM, KNN, and DNN classifiers.
Key contributions:

1) Full reproduction of nested CV results across all five classifiers
2) Feature ablation study revealing model-architecture dependent importance of Zenith angle vs NALLParticles — Zenith outperforms NALL only for the neural network, while distance and kernel-based models strongly prefer NALLParticles
3) Improved DNN implementation achieving 0.956 accuracy
4) ROC curve analysis for all classifiers
