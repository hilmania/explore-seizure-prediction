EEG Seizure Prediction Pipeline (TUSZ-derived EEG_NEW_16CHS_2)

This pipeline trains and evaluates preictal (1) vs interictal/normal (0) prediction using filtering, optional ICA, wrapper-based channel selection, chaos features, and multiple ML models.

Steps
- Preprocess: band-pass (0.5-40 Hz) + optional notch 60 Hz + optional ICA (per-epoch heuristic).
- Channel selection: Sequential forward selection (wrapper) to choose top-N channels.
- Feature extraction: Chaos features per selected channel (sample entropy, permutation/SVD entropy, fractal dimension, Lyapunov) + band powers.
- Models: Logistic Regression, SVM-RBF, RandomForest, GradientBoosting, KNN.
- Metrics: ROC-AUC, Accuracy, Specificity, Sensitivity, Precision, FPR, F1, F-beta.
- Alarm: triggers when probability >= threshold for K consecutive epochs.

Run
Use the provided Python environment detected by VS Code. Example:

python -m pipeline.run_pipeline --root /Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2 \
  --max_channels 8 --threshold 0.5 --min_consecutive 1 --beta 2.0

Outputs go to ./outputs/ by default.

Notes
- Assumes data arrays shaped (n_epochs, n_channels=16, n_samples=1280). Adjust config in `pipeline/config.py` if needed (e.g., sampling freq).
- ICA is heuristic and can be slow; disable via `--no_ica` to speed up.
- For class imbalance, models use class_weight='balanced'.
