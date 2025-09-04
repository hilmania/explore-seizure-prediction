import os
import argparse
import time
import json
import itertools
from typing import Dict, Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_split, describe
from .preprocess import preprocess_epochs
from .channel_select import sfs_channels
from .stats import (bootstrap_metric_ci, kruskal_wallis_test, surrogate_analysis,
                   mcnemar_test, permutation_test_metric, feature_impact_analysis,
                   channel_significance_test)
from sklearn.linear_model import LogisticRegression
from .features import extract_features
from .extra_features import extract_extra_features
from .feature_selection import AdvancedFeatureSelector, optimize_feature_combination
from .models import build_models
from .metrics import compute_metrics

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    RandomOverSampler = None
    RandomUnderSampler = None

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def run_single(cfg: PipelineConfig, root: str, params: Dict[str, Any]):
    # load data
    X_train, y_train = load_split(root, 'train')
    X_val, y_val = load_split(root, 'eval')

    # optional limits
    if params.get('limit_train'):
        X_train, y_train = X_train[: params['limit_train']], y_train[: params['limit_train']]
    if params.get('limit_eval'):
        X_val, y_val = X_val[: params['limit_eval']], y_val[: params['limit_eval']]

    # preprocess
    Xp_train = preprocess_epochs(X_train, cfg.preprocess)
    Xp_val = preprocess_epochs(X_val, cfg.preprocess)

    # channel selection with requested k (use quick logistic reg wrapper)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    sel = sfs_channels(Xp_train, y_train, clf, max_channels=params['max_channels'], scoring=cfg.channel_select.scoring, cv_folds=cfg.channel_select.cv_folds, step=cfg.channel_select.step)
    Xp_train_r = Xp_train[:, sel, :]
    Xp_val_r = Xp_val[:, sel, :]

    # features
    F_train = extract_features(Xp_train_r, cfg.features)
    F_val = extract_features(Xp_val_r, cfg.features)

    # extra features
    if params.get('add_time') or params.get('add_stft'):
        Fe_train, _ = extract_extra_features(Xp_train_r, sfreq=cfg.preprocess.sfreq, use_time=params.get('add_time', False), use_stft=params.get('add_stft', False))
        Fe_val, _ = extract_extra_features(Xp_val_r, sfreq=cfg.preprocess.sfreq, use_time=params.get('add_time', False), use_stft=params.get('add_stft', False))
        if Fe_train.size:
            F_train = np.hstack([F_train, Fe_train])
        if Fe_val.size:
            F_val = np.hstack([F_val, Fe_val])

    # record counts before
    if F_train.size:
        uniq_b, cnts_b = np.unique(y_train, return_counts=True)
        train_counts_before = {int(k): int(v) for k, v in zip(uniq_b.tolist(), cnts_b.tolist())}
    else:
        train_counts_before = {}

    # balancing
    if params.get('balance') and params['balance'] != 'none':
        if RandomOverSampler is None or RandomUnderSampler is None:
            pass
        else:
            if params['balance'] == 'oversample':
                ros = RandomOverSampler(random_state=cfg.model.random_state)
                F_train, y_train = ros.fit_resample(F_train, y_train)
            elif params['balance'] == 'undersample':
                rus = RandomUnderSampler(random_state=cfg.model.random_state)
                F_train, y_train = rus.fit_resample(F_train, y_train)

    if F_train.size:
        uniq_a, cnts_a = np.unique(y_train, return_counts=True)
        train_counts_after = {int(k): int(v) for k, v in zip(uniq_a.tolist(), cnts_a.tolist())}
    else:
        train_counts_after = {}

    # impute before PCA if needed
    if params.get('pca') and params['pca'] > 0:
        imp = SimpleImputer(strategy='median')
        F_train = imp.fit_transform(F_train)
        F_val = imp.transform(F_val)
        pca = PCA(n_components=params['pca'], random_state=cfg.model.random_state)
        F_train = pca.fit_transform(F_train)
        F_val = pca.transform(F_val)
    else:
        # ensure no-nans to satisfy some estimators when computing decision_function etc.
        imp = SimpleImputer(strategy='median')
        F_train = imp.fit_transform(F_train)
        F_val = imp.transform(F_val)

    # build models
    models = build_models(cfg.model.random_state, cfg.model.class_weight)

    run_results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(F_train, y_train)
        if hasattr(model[-1], 'predict_proba'):
            y_prob = model.predict_proba(F_val)[:, 1]
        else:
            s = model.decision_function(F_val)
            s = s.ravel() if s.ndim > 1 else s
            s_min, s_max = float(np.min(s)), float(np.max(s))
            y_prob = (s - s_min) / (s_max - s_min + 1e-8)

        # moving average
        if params.get('post_ma') and params['post_ma'] > 1:
            y_prob = pd.Series(y_prob).rolling(params['post_ma'], min_periods=1).mean().to_numpy()

        # threshold -> preds
        y_pred = (y_prob >= params.get('threshold', 0.5)).astype(int)

        # k-of-n voting
        if params.get('post_kofn'):
            try:
                K, N = map(int, params['post_kofn'].split(','))
                from collections import deque
                win = deque(maxlen=N)
                y_pred_k = np.zeros_like(y_pred)
                for i, v in enumerate(y_pred):
                    win.append(v)
                    if len(win) == N and sum(win) >= K:
                        y_pred_k[i] = 1
                y_pred = y_pred_k
            except Exception:
                pass

        metrics = compute_metrics(y_val, y_prob, y_pred, beta=params.get('beta', 2.0))
        t1 = time.time()
        row = {
            'model': name,
            'time_s': t1 - t0,
            'n_train': int(F_train.shape[0]),
            'n_eval': int(F_val.shape[0]),
            'train_counts_before': train_counts_before,
            'train_counts_after': train_counts_after,
        }
        row.update({f'metric_{k}': float(v) for k, v in metrics.items()})
        run_results.append(row)

    return sel, run_results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument('--max_channels_list', type=str, default='4', help='Comma separated channel counts to try')
    parser.add_argument('--balance_list', type=str, default='none', help='Comma separated balance strategies: none,oversample,undersample')
    parser.add_argument('--add_time', action='store_true')
    parser.add_argument('--add_stft', action='store_true')
    parser.add_argument('--pca', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--post_ma', type=int, default=0)
    parser.add_argument('--post_kofn', type=str, default='')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--limit_train', type=int, default=0)
    parser.add_argument('--limit_eval', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--export_dir', type=str, default='outputs')
    parser.add_argument('--no_ica', action='store_true', help='Disable ICA preprocessing during experiments to speed up runs')
    parser.add_argument('--stats', action='store_true', help='Run statistical validation tests (bootstrap CI, surrogate analysis, model comparisons)')
    parser.add_argument('--stats_metric', type=str, default='roc_auc', help='Primary metric for statistical tests')
    parser.add_argument('--n_boot', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--n_surrogates', type=int, default=1000, help='Number of surrogate samples')
    parser.add_argument('--feature_significance', action='store_true', help='Test significance of individual features')
    parser.add_argument('--channel_significance', action='store_true', help='Test significance of individual channels')
    parser.add_argument('--feature_test_type', type=str, default='permutation', choices=['permutation', 'ablation'], help='Type of feature importance test')

    # Feature selection arguments
    parser.add_argument('--feature_selection', action='store_true', help='Enable advanced feature selection')
    parser.add_argument('--fs_method', type=str, default='ensemble',
                       choices=['univariate', 'rfe', 'model', 'sfs', 'ensemble'],
                       help='Feature selection method')
    parser.add_argument('--fs_k', type=int, default=20, help='Number of features to select')
    parser.add_argument('--fs_estimator', type=str, default='rf',
                       choices=['rf', 'svm', 'logreg'],
                       help='Estimator for feature selection')
    parser.add_argument('--optimize_k_features', action='store_true',
                       help='Optimize number of features for best performance')
    parser.add_argument('--fs_k_range', type=str, default='10,50',
                       help='Range for optimizing number of features (min,max)')

    args = parser.parse_args(argv)

    cfg = PipelineConfig()
    if args.no_ica:
        cfg.preprocess.do_ica = False

    max_chs = [int(x) for x in args.max_channels_list.split(',') if x.strip()]
    balances = [x for x in args.balance_list.split(',') if x.strip()]

    os.makedirs(args.export_dir, exist_ok=True)
    all_rows = []
    for repeat in range(args.repeats):
        for max_ch, bal in itertools.product(max_chs, balances):
            params = {
                'max_channels': max_ch,
                'balance': bal,
                'add_time': args.add_time,
                'add_stft': args.add_stft,
                'pca': args.pca,
                'threshold': args.threshold,
                'post_ma': args.post_ma,
                'post_kofn': args.post_kofn,
                'beta': args.beta,
                'limit_train': args.limit_train,
                'limit_eval': args.limit_eval,
            }
            print('[exp] repeat', repeat, 'max_ch', max_ch, 'balance', bal)
            sel, rows = run_single(cfg, args.root, params)
            for r in rows:
                meta = {
                    'repeat': repeat,
                    'max_channels': max_ch,
                    'selected_channels': sel,
                    'balance': bal,
                    'add_time': args.add_time,
                    'add_stft': args.add_stft,
                    'pca': args.pca,
                    'post_ma': args.post_ma,
                    'post_kofn': args.post_kofn,
                }
                entry = {**meta, **r}
                all_rows.append(entry)

    # persist
    df = pd.json_normalize(all_rows)
    csvp = os.path.join(args.export_dir, 'experiments_results.csv')
    jpath = os.path.join(args.export_dir, 'experiments_results.json')
    df.to_csv(csvp, index=False)
    with open(jpath, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print('[exp] saved', csvp, jpath)

    # Statistical validation if requested
    if args.stats and len(all_rows) > 0:
        print('[stats] Running statistical validation...')
        stats_results = run_statistical_validation(all_rows, args)

        # Save statistical results
        stats_path = os.path.join(args.export_dir, 'statistical_validation.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2)
        print('[stats] saved', stats_path)

    # Feature significance analysis if requested
    if (args.feature_significance or args.channel_significance) and len(all_rows) > 0:
        print('[feature-stats] Running feature significance analysis...')
        feature_results = run_feature_significance_analysis(args)

        # Save feature significance results
        feature_stats_path = os.path.join(args.export_dir, 'feature_significance.json')
        with open(feature_stats_path, 'w') as f:
            json.dump(feature_results, f, indent=2)
        print('[feature-stats] saved', feature_stats_path)

    # Feature selection optimization if requested
    if args.feature_selection or args.optimize_k_features:
        print('[feature-selection] Running feature selection optimization...')
        fs_results = run_feature_selection_optimization(args)

        # Save feature selection results
        fs_path = os.path.join(args.export_dir, 'feature_selection.json')
        with open(fs_path, 'w') as f:
            json.dump(fs_results, f, indent=2)
        print('[feature-selection] saved', fs_path)


def run_feature_significance_analysis(args) -> dict:
    """Run feature and channel significance analysis."""
    from .data import load_split
    from .preprocess import preprocess_epochs
    from .features import extract_features
    from .extra_features import extract_extra_features
    from .config import PipelineConfig

    cfg = PipelineConfig()
    if args.no_ica:
        cfg.preprocess.do_ica = False

    # Load and preprocess data
    X_train, y_train = load_split(args.root, 'train')
    if args.limit_train and args.limit_train > 0:
        X_train, y_train = X_train[:args.limit_train], y_train[:args.limit_train]

    Xp_train = preprocess_epochs(X_train, cfg.preprocess)

    results = {
        'config': {
            'stats_metric': args.stats_metric,
            'feature_test_type': args.feature_test_type,
            'n_train_samples': len(y_train)
        }
    }

    # Channel significance test
    if args.channel_significance:
        print('[feature-stats] Testing channel significance...')
        try:
            # Generate channel names
            if hasattr(args, 'ch_names') and args.ch_names:
                ch_names = [s.strip() for s in args.ch_names.split(',')][:X_train.shape[1]]
            else:
                ch_names = [f'Ch_{i:02d}' for i in range(X_train.shape[1])]

            channel_results = channel_significance_test(
                Xp_train, y_train,
                channel_names=ch_names,
                metric_name=args.stats_metric,
                n_permutations=min(500, args.n_surrogates)
            )
            results['channel_significance'] = channel_results

            # Print significant channels
            sig_channels = [ch for ch, res in channel_results['channels'].items()
                          if res.get('significant', False)]
            print(f"[feature-stats] Significant channels ({len(sig_channels)}): {sig_channels}")

        except Exception as e:
            results['channel_significance'] = {'error': str(e)}

    # Feature significance test
    if args.feature_significance:
        print('[feature-stats] Testing feature significance...')
        try:
            # Use a sample of the data to get features for analysis
            sample_size = min(1000, len(Xp_train))
            idx_sample = np.random.choice(len(Xp_train), sample_size, replace=False)

            # Extract features from all channels
            F_sample = extract_features(Xp_train[idx_sample], cfg.features)

            # Add extra features if requested
            if args.add_time or args.add_stft:
                Fe_extra, _ = extract_extra_features(
                    Xp_train[idx_sample],
                    sfreq=cfg.preprocess.sfreq,
                    use_time=args.add_time,
                    use_stft=args.add_stft
                )
                if Fe_extra.size:
                    F_sample = np.hstack([F_sample, Fe_extra])

            # Generate feature names
            n_ch = Xp_train.shape[1]
            base_features = ['sampen', 'perm_ent', 'svd_ent', 'higuchi_fd', 'petrosian_fd',
                           'dfa', 'hurst', 'delta_power', 'theta_power', 'alpha_power', 'beta_power']
            feature_names = []
            for ch in range(n_ch):
                for feat in base_features:
                    feature_names.append(f'Ch{ch:02d}_{feat}')

            if args.add_time:
                time_features = ['mean', 'std', 'skewness', 'kurtosis', 'peak2peak', 'rms']
                for ch in range(n_ch):
                    for feat in time_features:
                        feature_names.append(f'Ch{ch:02d}_time_{feat}')

            if args.add_stft:
                stft_features = ['stft_delta', 'stft_theta', 'stft_alpha', 'stft_beta']
                for ch in range(n_ch):
                    for feat in stft_features:
                        feature_names.append(f'Ch{ch:02d}_{feat}')

            # Trim feature names to match actual features
            feature_names = feature_names[:F_sample.shape[1]]

            feature_results = feature_impact_analysis(
                F_sample, y_train[idx_sample],
                feature_names=feature_names,
                metric_name=args.stats_metric,
                n_permutations=min(200, args.n_surrogates),
                test_type=args.feature_test_type
            )
            results['feature_significance'] = feature_results

            # Print top significant features
            if 'features' in feature_results:
                sig_features = [(name, res) for name, res in feature_results['features'].items()
                              if res.get('significant', False)]
                sig_features.sort(key=lambda x: x[1]['importance'], reverse=True)
                top_features = sig_features[:10]
                print(f"[feature-stats] Top significant features ({len(sig_features)} total):")
                for name, res in top_features:
                    print(f"  {name}: importance={res['importance']:.4f}, p={res['p_value']:.4f}")

        except Exception as e:
            results['feature_significance'] = {'error': str(e)}

    return results


def run_feature_selection_optimization(args) -> dict:
    """
    Run feature selection optimization to find best feature combinations
    """
    from .data import load_split
    from .preprocess import preprocess_epochs
    from .features import extract_features
    from .extra_features import extract_extra_features
    from .config import PipelineConfig
    from .channel_select import sfs_channels
    from sklearn.linear_model import LogisticRegression

    print("[feature-selection] Starting feature selection optimization...")

    cfg = PipelineConfig()
    if args.no_ica:
        cfg.preprocess.do_ica = False

    # Load and preprocess data
    X_train, y_train = load_split(args.root, 'train')
    if args.limit_train and args.limit_train > 0:
        X_train, y_train = X_train[:args.limit_train], y_train[:args.limit_train]

    Xp_train = preprocess_epochs(X_train, cfg.preprocess)

    # Channel selection (use first max_channels setting)
    max_chs = [int(x) for x in args.max_channels_list.split(',') if x.strip()]
    max_channels = max_chs[0] if max_chs else 4

    selected_channels = sfs_channels(
        Xp_train, y_train,
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=cfg.model.random_state),
        max_channels=max_channels
    )

    # Reduce to selected channels
    Xp_train_selected = Xp_train[:, selected_channels, :]

    # Extract features
    F_train = extract_features(Xp_train_selected, cfg.features)

    # Add extra features if requested
    if args.add_time or args.add_stft:
        Fe_train, feature_names_extra = extract_extra_features(
            Xp_train_selected,
            sfreq=cfg.preprocess.sfreq,
            use_time=args.add_time,
            use_stft=args.add_stft
        )
        F_train = np.concatenate([F_train, Fe_train], axis=1)

    # Create feature names
    ch_names = [f"Ch{i:02d}" for i in selected_channels]
    feature_names = []

    # Core chaos features
    for ch in ch_names:
        for feat in ['sampen', 'perm_ent', 'svd_ent', 'higuchi_fd', 'petrosian_fd',
                     'dfa', 'hurst', 'bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta']:
            feature_names.append(f'{ch}_{feat}')

    # Extra features
    if args.add_time:
        for ch in ch_names:
            for feat in ['mean', 'std', 'skew', 'kurt', 'ptp', 'rms']:
                feature_names.append(f'{ch}_{feat}')

    if args.add_stft:
        for ch in ch_names:
            for feat in ['stft_bp_delta', 'stft_bp_theta', 'stft_bp_alpha', 'stft_bp_beta']:
                feature_names.append(f'{ch}_{feat}')

    # Trim feature names to match actual features
    feature_names = feature_names[:F_train.shape[1]]

    print(f"[feature-selection] Total features available: {F_train.shape[1]}")
    print(f"[feature-selection] Selected channels: {selected_channels}")

    # Initialize feature selector
    selector = AdvancedFeatureSelector(random_state=cfg.model.random_state)

    results = {
        'config': {
            'method': args.fs_method,
            'k_features': args.fs_k,
            'estimator': args.fs_estimator,
            'total_features': F_train.shape[1],
            'selected_channels': selected_channels,
            'n_samples': F_train.shape[0]
        },
        'selection_results': {}
    }

    try:
        # Run feature selection based on method
        if args.fs_method == 'univariate':
            for score_func in ['f_classif', 'mutual_info']:
                print(f"[feature-selection] Running univariate selection with {score_func}...")
                result = selector.univariate_selection(
                    F_train, y_train, feature_names, score_func, args.fs_k
                )
                results['selection_results'][f'univariate_{score_func}'] = {
                    'selected_features': result['selected_names'],
                    'selected_indices': result['selected_indices'].tolist(),
                    'scores': result['scores'].tolist()
                }

        elif args.fs_method == 'rfe':
            print(f"[feature-selection] Running RFE with {args.fs_estimator}...")
            result = selector.recursive_feature_elimination(
                F_train, y_train, feature_names, args.fs_estimator, args.fs_k, cv_folds=3, use_cv=True
            )
            results['selection_results']['rfe'] = {
                'selected_features': result['selected_names'],
                'selected_indices': result['selected_indices'].tolist(),
                'rankings': result['rankings'].tolist(),
                'n_features_selected': result['n_features_selected']
            }

        elif args.fs_method == 'model':
            print(f"[feature-selection] Running model-based selection with {args.fs_estimator}...")
            result = selector.model_based_selection(
                F_train, y_train, feature_names, args.fs_estimator, 'median'
            )
            results['selection_results']['model'] = {
                'selected_features': result['selected_names'],
                'selected_indices': result['selected_indices'].tolist(),
                'importances': result['importances'].tolist(),
                'threshold': float(result['threshold_used'])
            }

        elif args.fs_method == 'sfs':
            print(f"[feature-selection] Running sequential forward selection with {args.fs_estimator}...")
            result = selector.sequential_forward_selection(
                F_train, y_train, feature_names, args.fs_estimator, args.fs_k
            )
            results['selection_results']['sfs'] = {
                'selected_features': result['selected_names'],
                'selected_indices': result['selected_indices'].tolist(),
                'scores_history': result['scores_history']
            }

        elif args.fs_method == 'ensemble':
            print("[feature-selection] Running ensemble feature selection...")
            methods = ['univariate_f_classif', 'rfe_rf', 'model_rf']
            result = selector.ensemble_feature_selection(
                F_train, y_train, feature_names, methods, args.fs_k, voting_threshold=0.3
            )
            results['selection_results']['ensemble'] = {
                'selected_features': result['selected_names'],
                'selected_indices': result['selected_indices'].tolist(),
                'feature_votes': result['feature_votes'].tolist(),
                'voting_threshold': result['voting_threshold']
            }

            # Include individual method results
            for method, method_result in result['method_results'].items():
                results['selection_results'][f'ensemble_{method}'] = {
                    'selected_features': method_result['selected_names'],
                    'selected_indices': method_result['selected_indices'].tolist()
                }

        # Optimize number of features if requested
        if args.optimize_k_features:
            print("[feature-selection] Optimizing number of features...")
            k_min, k_max = map(int, args.fs_k_range.split(','))
            opt_result = optimize_feature_combination(
                F_train, y_train, feature_names, args.fs_estimator,
                (k_min, k_max), step=5, cv_folds=3
            )
            results['optimization'] = {
                'optimal_n_features': opt_result['optimal_n_features'],
                'optimal_score': opt_result['optimal_score'],
                'n_features_tested': opt_result['n_features_list'],
                'scores_mean': opt_result['scores_mean'],
                'scores_std': opt_result['scores_std']
            }
            print(f"[feature-selection] Optimal number of features: {opt_result['optimal_n_features']} "
                  f"(score: {opt_result['optimal_score']:.4f})")

        # Print summary of selected features
        for method, result in results['selection_results'].items():
            if 'selected_features' in result:
                n_selected = len(result['selected_features'])
                print(f"[feature-selection] {method}: selected {n_selected} features")
                if n_selected <= 10:
                    print(f"  Features: {', '.join(result['selected_features'])}")
                else:
                    print(f"  Top 5 features: {', '.join(result['selected_features'][:5])}")

    except Exception as e:
        results['error'] = str(e)
        print(f"[feature-selection] Error: {e}")

    return results


def run_statistical_validation(all_rows: list, args) -> dict:
    """Run statistical validation tests on experiment results."""
    stats_results = {
        'config': {
            'stats_metric': args.stats_metric,
            'n_boot': args.n_boot,
            'n_surrogates': args.n_surrogates
        },
        'tests': {}
    }

    # Group results by model for analysis
    models_data = {}
    for row in all_rows:
        model_name = row['model']
        if model_name not in models_data:
            models_data[model_name] = []

        # Extract metric value
        metric_key = f'metric_{args.stats_metric}'
        if metric_key in row:
            models_data[model_name].append(row[metric_key])

    # 1. Kruskal-Wallis test for comparing all models
    if len(models_data) >= 3:
        try:
            groups = list(models_data.values())
            group_names = list(models_data.keys())
            kw_result = kruskal_wallis_test(groups, group_names)
            stats_results['tests']['kruskal_wallis'] = kw_result
            print(f"[stats] Kruskal-Wallis test: H={kw_result['statistic']:.3f}, p={kw_result['p_value']:.4f}")
        except Exception as e:
            stats_results['tests']['kruskal_wallis'] = {'error': str(e)}

    # 2. Bootstrap confidence intervals for each model
    bootstrap_results = {}
    for model_name, metrics in models_data.items():
        if len(metrics) > 1:  # Need multiple samples for bootstrap
            try:
                # For bootstrap CI, we need the actual y_true, y_prob, y_pred
                # Since we don't store these in experiments, we'll bootstrap the metric values directly
                ci_result = bootstrap_metric_values(metrics, args.n_boot, 0.95)
                bootstrap_results[model_name] = ci_result
                print(f"[stats] {model_name} {args.stats_metric} CI: [{ci_result['ci_lo']:.3f}, {ci_result['ci_hi']:.3f}]")
            except Exception as e:
                bootstrap_results[model_name] = {'error': str(e)}
    stats_results['tests']['bootstrap_ci'] = bootstrap_results

    # 3. Pairwise model comparisons (permutation tests on metric values)
    pairwise_results = {}
    model_names = list(models_data.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model_a, model_b = model_names[i], model_names[j]
            metrics_a, metrics_b = models_data[model_a], models_data[model_b]

            # Only compare if we have same number of samples
            min_len = min(len(metrics_a), len(metrics_b))
            if min_len > 1:
                try:
                    # Simple paired test on metric differences
                    diff_result = paired_metric_test(metrics_a[:min_len], metrics_b[:min_len])
                    pair_key = f"{model_a}_vs_{model_b}"
                    pairwise_results[pair_key] = diff_result
                    print(f"[stats] {pair_key}: diff={diff_result['mean_diff']:.4f}, p={diff_result['p_value']:.4f}")
                except Exception as e:
                    pairwise_results[f"{model_a}_vs_{model_b}"] = {'error': str(e)}
    stats_results['tests']['pairwise_comparisons'] = pairwise_results

    # 4. Summary statistics
    summary = {}
    for model_name, metrics in models_data.items():
        if metrics:
            summary[model_name] = {
                'mean': float(np.mean(metrics)),
                'std': float(np.std(metrics)),
                'median': float(np.median(metrics)),
                'min': float(np.min(metrics)),
                'max': float(np.max(metrics)),
                'n_samples': len(metrics)
            }
    stats_results['summary'] = summary

    return stats_results


def bootstrap_metric_values(values: list, n_boot: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for a list of metric values."""
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        boot_sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, (1.0 - ci) / 2.0 * 100)
    hi = np.percentile(boot_means, (1.0 + ci) / 2.0 * 100)

    return {
        'mean': float(np.mean(values)),
        'ci_lo': float(lo),
        'ci_hi': float(hi)
    }


def paired_metric_test(values_a: list, values_b: list, n_perm: int = 1000) -> dict:
    """Permutation test for paired differences in metric values."""
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) == 0:
        return {'mean_diff': float('nan'), 'p_value': float('nan')}

    observed_diff = np.mean(a - b)

    # Permutation test
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        # Randomly swap elements between a and b
        swap_mask = rng.random(len(a)) < 0.5
        perm_a = np.where(swap_mask, b, a)
        perm_b = np.where(swap_mask, a, b)
        perm_diff = np.mean(perm_a - perm_b)

        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_perm + 1)

    return {
        'mean_diff': float(observed_diff),
        'p_value': float(p_value)
    }


if __name__ == '__main__':
    main()
