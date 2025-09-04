"""Statistical validation helpers for model predictions.

Provides:
- bootstrap_metric_ci: bootstrap confidence intervals for any metric available in pipeline.metrics.compute_metrics
- permutation_test_metric: paired permutation test between two sets of scores/predictions
- mcnemar_test: McNemar's test for paired binary predictions
- kruskal_wallis_test: Kruskal-Wallis H-test for comparing multiple independent groups
- surrogate_analysis: surrogate data analysis to test significance against chance level
- feature_impact_analysis: test significance of individual features on model performance
- channel_significance_test: test predictive power of individual EEG channels

These are lightweight utilities intended for experiment-level validation.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Tuple, Sequence, List, Union
from scipy.stats import norm, chi2, kruskal

from .metrics import compute_metrics
from sklearn.metrics import roc_auc_score, accuracy_score


def _safe_metric_call(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, metric_name: str, beta: float = 2.0):
    m = compute_metrics(y_true, y_prob, y_pred, beta=beta)
    return m.get(metric_name)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = 'roc_auc',
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    beta: float = 2.0,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns dict with keys: 'metric', 'ci_lo', 'ci_hi', 'mean'.
    metric_name should be one of the keys returned by `compute_metrics`, e.g. 'roc_auc','accuracy','f1', etc.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = _safe_metric_call(y_true[idx], y_prob[idx], y_pred[idx], metric_name, beta=beta)
        except Exception:
            v = np.nan
        vals.append(v)
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {'metric': float('nan'), 'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}
    lo = np.percentile(arr, (1.0 - ci) / 2.0 * 100)
    hi = np.percentile(arr, (1.0 + ci) / 2.0 * 100)
    return {'metric': float(_safe_metric_call(y_true, y_prob, y_pred, metric_name, beta=beta)), 'mean': float(arr.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi)}


def permutation_test_metric(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str = 'roc_auc',
    n_perm: int = 1000,
    seed: int = 0,
    beta: float = 2.0,
) -> Dict[str, float]:
    """Paired permutation test for difference in metric between two models.

    - scores_a / scores_b: can be probabilities (for metrics needing y_prob) or binary preds for label-based metrics.
    - metric: 'roc_auc' or 'accuracy' or any metric supported via compute_metrics when both prob and pred provided. For simplicity, when metric=='roc_auc' we use roc_auc_score on scores.

    Returns {'observed_diff': float, 'p_value': float}
    """
    rng = np.random.default_rng(seed)
    assert len(y_true) == len(scores_a) == len(scores_b)
    # helper to compute metric value for a score vector
    def metric_val(scores):
        if metric == 'roc_auc':
            try:
                return roc_auc_score(y_true, scores)
            except Exception:
                return float('nan')
        elif metric == 'accuracy':
            # if scores are probabilities, threshold at 0.5
            preds = (scores >= 0.5).astype(int)
            return float(accuracy_score(y_true, preds))
        else:
            # fallback: require both prob+pred; here we try to create preds by thresholding
            preds = (scores >= 0.5).astype(int)
            m = compute_metrics(y_true, scores, preds, beta=beta)
            return float(m.get(metric, float('nan')))

    obs_a = metric_val(scores_a)
    obs_b = metric_val(scores_b)
    observed_diff = obs_a - obs_b

    count = 0
    n = len(y_true)
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, scores_b, scores_a)
        perm_b = np.where(swap, scores_a, scores_b)
        pa = metric_val(perm_a)
        pb = metric_val(perm_b)
        if (pa - pb) >= abs(observed_diff):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {'observed_diff': float(observed_diff), 'p_value': float(p_value)}


def mcnemar_test(y_true: np.ndarray, pred_a: Sequence[int], pred_b: Sequence[int]) -> Dict[str, float]:
    """Perform McNemar's test for two paired binary classifiers.

    Returns {'b':int,'c':int,'chi2':float,'p_value':float}
    where b = A correct & B incorrect, c = A incorrect & B correct.
    Uses continuity-corrected statistic when b+c > 0.
    """
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    y = np.asarray(y_true).astype(int)
    a_correct = pred_a == y
    b_correct = pred_b == y
    b = int(np.sum(np.logical_and(a_correct, ~b_correct)))
    c = int(np.sum(np.logical_and(~a_correct, b_correct)))
    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': 0.0, 'p_value': 1.0}
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p = 1.0 - chi2.cdf(chi2_stat, df=1)
    return {'b': b, 'c': c, 'chi2': float(chi2_stat), 'p_value': float(p)}


def kruskal_wallis_test(
    groups: List[Union[np.ndarray, Sequence]],
    group_names: List[str] = None
) -> Dict[str, Union[float, str, List]]:
    """Perform Kruskal-Wallis H-test for comparing multiple independent groups.

    Non-parametric test for comparing 3+ groups when assumptions of ANOVA are violated.

    Args:
        groups: List of arrays, each containing metric values for one group (e.g., different models)
        group_names: Optional names for groups (defaults to 'Group_0', 'Group_1', ...)

    Returns:
        Dict with 'statistic', 'p_value', 'groups', 'group_names', 'group_sizes'
    """
    if group_names is None:
        group_names = [f'Group_{i}' for i in range(len(groups))]

    # Convert to arrays and filter out NaN values
    clean_groups = []
    clean_names = []
    group_sizes = []

    for i, group in enumerate(groups):
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)
            clean_names.append(group_names[i])
            group_sizes.append(len(arr))

    if len(clean_groups) < 2:
        return {
            'statistic': float('nan'),
            'p_value': float('nan'),
            'groups': len(clean_groups),
            'group_names': clean_names,
            'group_sizes': group_sizes
        }

    try:
        statistic, p_value = kruskal(*clean_groups)
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'groups': len(clean_groups),
            'group_names': clean_names,
            'group_sizes': group_sizes
        }
    except Exception as e:
        return {
            'statistic': float('nan'),
            'p_value': float('nan'),
            'groups': len(clean_groups),
            'group_names': clean_names,
            'group_sizes': group_sizes,
            'error': str(e)
        }


def surrogate_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = 'roc_auc',
    n_surrogates: int = 1000,
    surrogate_method: str = 'shuffle',
    seed: int = 0,
    beta: float = 2.0
) -> Dict[str, float]:
    """Surrogate data analysis to test if observed metric is significantly different from chance.

    Creates surrogate datasets by shuffling labels or predictions and compares observed metric
    against distribution of surrogate metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted binary labels
        metric_name: Metric to test (e.g., 'roc_auc', 'accuracy')
        n_surrogates: Number of surrogate datasets to generate
        surrogate_method: 'shuffle_labels' (shuffle y_true) or 'shuffle_preds' (shuffle predictions)
        seed: Random seed
        beta: Beta parameter for F-beta score

    Returns:
        Dict with 'observed_metric', 'surrogate_mean', 'surrogate_std', 'p_value', 'z_score'
    """
    rng = np.random.default_rng(seed)

    # Calculate observed metric
    try:
        observed = _safe_metric_call(y_true, y_prob, y_pred, metric_name, beta=beta)
    except Exception:
        return {
            'observed_metric': float('nan'),
            'surrogate_mean': float('nan'),
            'surrogate_std': float('nan'),
            'p_value': float('nan'),
            'z_score': float('nan')
        }

    # Generate surrogate metrics
    surrogate_metrics = []
    for _ in range(n_surrogates):
        if surrogate_method == 'shuffle_labels':
            # Shuffle true labels
            y_surr = rng.permutation(y_true)
            try:
                surr_metric = _safe_metric_call(y_surr, y_prob, y_pred, metric_name, beta=beta)
            except Exception:
                surr_metric = np.nan
        elif surrogate_method == 'shuffle_preds':
            # Shuffle predictions
            y_prob_surr = rng.permutation(y_prob)
            y_pred_surr = rng.permutation(y_pred)
            try:
                surr_metric = _safe_metric_call(y_true, y_prob_surr, y_pred_surr, metric_name, beta=beta)
            except Exception:
                surr_metric = np.nan
        else:
            raise ValueError(f"Unknown surrogate_method: {surrogate_method}")

        surrogate_metrics.append(surr_metric)

    # Clean surrogate metrics
    surrogate_metrics = np.array(surrogate_metrics, dtype=float)
    surrogate_metrics = surrogate_metrics[~np.isnan(surrogate_metrics)]

    if len(surrogate_metrics) == 0:
        return {
            'observed_metric': float(observed),
            'surrogate_mean': float('nan'),
            'surrogate_std': float('nan'),
            'p_value': float('nan'),
            'z_score': float('nan')
        }

    # Calculate statistics
    surr_mean = float(np.mean(surrogate_metrics))
    surr_std = float(np.std(surrogate_metrics))

    # Calculate p-value (two-tailed test)
    if surr_std > 0:
        z_score = (observed - surr_mean) / surr_std
        # Count how many surrogates are as extreme as observed
        extreme_count = np.sum(np.abs(surrogate_metrics - surr_mean) >= np.abs(observed - surr_mean))
        p_value = extreme_count / len(surrogate_metrics)
    else:
        z_score = float('nan')
        p_value = 1.0 if np.abs(observed - surr_mean) < 1e-10 else 0.0

    return {
        'observed_metric': float(observed),
        'surrogate_mean': surr_mean,
        'surrogate_std': surr_std,
        'p_value': float(p_value),
        'z_score': float(z_score)
    }


def feature_impact_analysis(
    X_features: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str] = None,
    model=None,
    metric_name: str = 'roc_auc',
    n_permutations: int = 1000,
    test_type: str = 'permutation',
    seed: int = 42,
    beta: float = 2.0
) -> Dict[str, Dict]:
    """Analyze the significance of individual features on model performance.

    Tests whether each feature contributes significantly to model performance by:
    1. Feature permutation importance: shuffle each feature and measure performance drop
    2. Feature ablation: remove each feature and retrain model

    Args:
        X_features: Feature matrix (n_samples, n_features)
        y_true: True labels
        feature_names: Names of features (defaults to Feature_0, Feature_1, ...)
        model: Trained sklearn model or None (will use LogisticRegression)
        metric_name: Performance metric to analyze
        n_permutations: Number of permutations for significance testing
        test_type: 'permutation' or 'ablation'
        seed: Random seed
        beta: Beta parameter for F-beta score

    Returns:
        Dict with feature importance scores and p-values
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_features.shape[1])]

    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')

    rng = np.random.default_rng(seed)

    # Calculate baseline performance
    try:
        if test_type == 'permutation':
            # Train model on original data
            model.fit(X_features, y_true)
            if hasattr(model, 'predict_proba'):
                y_prob_baseline = model.predict_proba(X_features)[:, 1]
            else:
                y_prob_baseline = model.decision_function(X_features)
                y_prob_baseline = (y_prob_baseline - y_prob_baseline.min()) / (y_prob_baseline.max() - y_prob_baseline.min() + 1e-8)
            y_pred_baseline = (y_prob_baseline >= 0.5).astype(int)
            baseline_score = _safe_metric_call(y_true, y_prob_baseline, y_pred_baseline, metric_name, beta=beta)
        else:  # ablation
            baseline_score = np.mean(cross_val_score(model, X_features, y_true, cv=5, scoring='roc_auc' if metric_name == 'roc_auc' else 'accuracy'))
    except Exception as e:
        return {'error': f'Baseline calculation failed: {str(e)}'}

    results = {}

    for i, feature_name in enumerate(feature_names):
        try:
            if test_type == 'permutation':
                # Permutation importance
                importance_scores = []
                for _ in range(n_permutations):
                    X_perm = X_features.copy()
                    X_perm[:, i] = rng.permutation(X_perm[:, i])

                    if hasattr(model, 'predict_proba'):
                        y_prob_perm = model.predict_proba(X_perm)[:, 1]
                    else:
                        y_prob_perm = model.decision_function(X_perm)
                        y_prob_perm = (y_prob_perm - y_prob_perm.min()) / (y_prob_perm.max() - y_prob_perm.min() + 1e-8)
                    y_pred_perm = (y_prob_perm >= 0.5).astype(int)

                    perm_score = _safe_metric_call(y_true, y_prob_perm, y_pred_perm, metric_name, beta=beta)
                    importance_scores.append(baseline_score - perm_score)  # positive = feature is important

                importance_scores = np.array(importance_scores)
                importance_scores = importance_scores[~np.isnan(importance_scores)]

                if len(importance_scores) > 0:
                    mean_importance = float(np.mean(importance_scores))
                    std_importance = float(np.std(importance_scores))
                    # P-value: how often is the importance <= 0 (feature doesn't help)
                    p_value = float(np.mean(importance_scores <= 0))
                else:
                    mean_importance = float('nan')
                    std_importance = float('nan')
                    p_value = 1.0

            else:  # ablation
                # Feature ablation
                X_ablated = np.delete(X_features, i, axis=1)
                ablated_scores = cross_val_score(model, X_ablated, y_true, cv=5, scoring='roc_auc' if metric_name == 'roc_auc' else 'accuracy')
                ablated_score = np.mean(ablated_scores)

                mean_importance = float(baseline_score - ablated_score)
                std_importance = float(np.std(ablated_scores))
                # Simple t-test approximation
                if std_importance > 0:
                    t_stat = mean_importance / (std_importance / np.sqrt(len(ablated_scores)))
                    # Approximate p-value (two-tailed)
                    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
                else:
                    p_value = 1.0 if abs(mean_importance) < 1e-10 else 0.0

            results[feature_name] = {
                'importance': mean_importance,
                'std': std_importance,
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        except Exception as e:
            results[feature_name] = {
                'importance': float('nan'),
                'std': float('nan'),
                'p_value': 1.0,
                'significant': False,
                'error': str(e)
            }

    return {
        'baseline_score': float(baseline_score),
        'test_type': test_type,
        'metric': metric_name,
        'n_permutations': n_permutations if test_type == 'permutation' else None,
        'features': results
    }


def channel_significance_test(
    X_epochs: np.ndarray,
    y_true: np.ndarray,
    channel_names: List[str] = None,
    metric_name: str = 'roc_auc',
    n_permutations: int = 500,
    seed: int = 42
) -> Dict[str, Dict]:
    """Test significance of individual EEG channels for seizure prediction.

    For each channel, computes features and tests predictive power via permutation test.

    Args:
        X_epochs: Raw EEG data (n_epochs, n_channels, n_timepoints)
        y_true: True labels
        channel_names: Channel names (defaults to Ch_0, Ch_1, ...)
        metric_name: Performance metric
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dict with per-channel significance results
    """
    from .features import extract_features
    from .config import PipelineConfig
    from sklearn.linear_model import LogisticRegression

    if channel_names is None:
        channel_names = [f'Ch_{i}' for i in range(X_epochs.shape[1])]

    cfg = PipelineConfig()
    results = {}

    for i, ch_name in enumerate(channel_names):
        try:
            # Extract features for this channel only
            X_ch = X_epochs[:, [i], :]  # Keep channel dimension
            F_ch = extract_features(X_ch, cfg.features)

            # Skip if all features are NaN
            if np.all(np.isnan(F_ch)):
                results[ch_name] = {
                    'roc_auc': float('nan'),
                    'p_value': 1.0,
                    'significant': False,
                    'error': 'All features are NaN'
                }
                continue

            # Simple imputation
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            F_ch_clean = imputer.fit_transform(F_ch)

            # Train classifier
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(F_ch_clean, y_true)

            # Calculate observed performance
            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(F_ch_clean)[:, 1]
            else:
                y_prob = clf.decision_function(F_ch_clean)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

            if metric_name == 'roc_auc':
                observed_score = roc_auc_score(y_true, y_prob)
            else:
                y_pred = (y_prob >= 0.5).astype(int)
                observed_score = _safe_metric_call(y_true, y_prob, y_pred, metric_name)

            # Permutation test
            rng = np.random.default_rng(seed + i)
            perm_scores = []

            for _ in range(n_permutations):
                y_perm = rng.permutation(y_true)
                if metric_name == 'roc_auc':
                    try:
                        perm_score = roc_auc_score(y_perm, y_prob)
                    except:
                        perm_score = 0.5
                else:
                    y_pred = (y_prob >= 0.5).astype(int)
                    perm_score = _safe_metric_call(y_perm, y_prob, y_pred, metric_name)
                perm_scores.append(perm_score)

            perm_scores = np.array(perm_scores)
            perm_scores = perm_scores[~np.isnan(perm_scores)]

            if len(perm_scores) > 0:
                p_value = float(np.mean(perm_scores >= observed_score))
            else:
                p_value = 1.0

            results[ch_name] = {
                metric_name: float(observed_score),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_features': F_ch.shape[1]
            }

        except Exception as e:
            results[ch_name] = {
                metric_name: float('nan'),
                'p_value': 1.0,
                'significant': False,
                'error': str(e)
            }

    return {
        'metric': metric_name,
        'n_permutations': n_permutations,
        'channels': results
    }
