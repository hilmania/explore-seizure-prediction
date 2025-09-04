import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

from .config import PipelineConfig
from .data import load_split, describe
from .preprocess import preprocess_epochs
from .features import extract_features
from .extra_features import extract_extra_features
from .channel_select import sfs_channels
from .models import build_models
from .metrics import compute_metrics
# balancing samplers
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    RandomOverSampler = None
    RandomUnderSampler = None


def alarm_from_probs(
    y_prob: np.ndarray,
    threshold: float = 0.5,
    min_consecutive: int = 1,
    t_on: float | None = None,
    t_off: float | None = None,
    min_consecutive_on: int | None = None,
    min_consecutive_off: int = 1,
    refractory: int = 0,
):
    """
    Alarm with hysteresis and refractory period.

    - t_on/t_off: thresholds to turn alarm ON/OFF (hysteresis). Defaults to `threshold` if not given.
    - min_consecutive_on/min_consecutive_off: consecutive epochs needed to turn ON/OFF. If not given, `min_consecutive` used for ON.
    - refractory: after turning ON, disallow re-arming for N epochs after alarm turns OFF.

    Returns an int array of the same length as y_prob where 1 indicates alarm active.
    """
    if t_on is None:
        t_on = threshold
    if t_off is None:
        t_off = threshold
    if min_consecutive_on is None:
        min_consecutive_on = min_consecutive

    alarms = np.zeros_like(y_prob, dtype=int)
    state_on = False
    on_count = 0
    off_count = 0
    cooldown = 0  # refractory countdown

    for i, p in enumerate(y_prob):
        if not state_on:
            # decrease cooldown when off
            if cooldown > 0:
                cooldown -= 1
            if cooldown == 0:
                if p >= t_on:
                    on_count += 1
                else:
                    on_count = 0
                if on_count >= min_consecutive_on:
                    state_on = True
                    alarms[i] = 1
                    on_count = 0
                    # cooldown will be set when turning OFF
            # when off and still in cooldown, remain off
        else:
            # state ON
            alarms[i] = 1
            if p < t_off:
                off_count += 1
            else:
                off_count = 0
            if off_count >= min_consecutive_off:
                # turn off immediately at this epoch
                state_on = False
                alarms[i] = 0
                off_count = 0
                cooldown = refractory
    return alarms


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_consecutive', type=int, default=1)
    parser.add_argument('--max_channels', type=int, default=8)
    parser.add_argument('--no_ica', action='store_true')
    parser.add_argument('--add_time_features', action='store_true', help='Add time-domain stats')
    parser.add_argument('--add_stft_features', action='store_true', help='Add STFT bandpower features')
    parser.add_argument('--balance', type=str, default='none', choices=['none','undersample','oversample'], help='Data balancing strategy')
    parser.add_argument('--pca', type=int, default=0, help='Apply PCA to features and keep this many components (0=none)')
    parser.add_argument('--post_kofn', type=str, default='', help='k_of_n voting, format K,N (e.g., 3,5)')
    parser.add_argument('--post_ma', type=int, default=0, help='moving-average window on y_prob before thresholding')
    parser.add_argument('--export_dir', type=str, default='outputs')
    parser.add_argument('--ch_names', type=str, default='', help='Comma-separated channel names in order; if empty, defaults to CH00..')
    parser.add_argument('--ch_names_file', type=str, default='', help='Path to a text file with one channel name per line')
    parser.add_argument('--limit_train', type=int, default=0, help='Limit train epochs for quick run (0=all)')
    parser.add_argument('--limit_eval', type=int, default=0, help='Limit eval epochs for quick run (0=all)')
    parser.add_argument('--t_on', type=float, default=None, help='Threshold to turn alarm ON (default: threshold)')
    parser.add_argument('--t_off', type=float, default=None, help='Threshold to turn alarm OFF (default: threshold)')
    parser.add_argument('--min_consecutive_on', type=int, default=None, help='Consecutive epochs to turn ON (default: min_consecutive)')
    parser.add_argument('--min_consecutive_off', type=int, default=1, help='Consecutive epochs to turn OFF (default: 1)')
    parser.add_argument('--refractory', type=int, default=0, help='Cooldown epochs after alarm turns OFF before it can re-arm')
    ns = parser.parse_args(args)

    cfg = PipelineConfig()
    cfg.channel_select.max_channels = ns.max_channels
    if ns.no_ica:
        cfg.preprocess.do_ica = False

    # Load data
    X_train, y_train = load_split(ns.root, 'train')
    X_val, y_val = load_split(ns.root, 'eval')
    print('[data] train', describe(X_train, y_train))
    print('[data] eval ', describe(X_val, y_val))

    # Preprocess
    if ns.limit_train and ns.limit_train > 0:
        X_train, y_train = X_train[:ns.limit_train], y_train[:ns.limit_train]
    if ns.limit_eval and ns.limit_eval > 0:
        X_val, y_val = X_val[:ns.limit_eval], y_val[:ns.limit_eval]

    Xp_train = preprocess_epochs(X_train, cfg.preprocess)
    Xp_val = preprocess_epochs(X_val, cfg.preprocess)

    # Channel selection (wrapper) using a simple classifier (logreg)
    from sklearn.linear_model import LogisticRegression
    sel = sfs_channels(Xp_train, y_train, LogisticRegression(max_iter=1000, class_weight='balanced'), max_channels=cfg.channel_select.max_channels, scoring=cfg.channel_select.scoring, cv_folds=cfg.channel_select.cv_folds, step=cfg.channel_select.step)
    # Resolve channel names
    def build_channel_names(n_ch: int):
        names = None
        if ns.ch_names_file and os.path.isfile(ns.ch_names_file):
            with open(ns.ch_names_file, 'r') as f:
                names = [ln.strip() for ln in f.readlines() if ln.strip()]
        elif ns.ch_names:
            names = [s.strip() for s in ns.ch_names.split(',') if s.strip()]
        if not names or len(names) != n_ch:
            names = [f'CH{i:02d}' for i in range(n_ch)]
        return names
    all_names = build_channel_names(X_train.shape[1])
    sel_names = [all_names[i] for i in sel]
    print('[channel-select] selected indices:', sel)
    print('[channel-select] selected names  :', sel_names)

    # Reduce channels
    Xp_train = Xp_train[:, sel, :]
    Xp_val = Xp_val[:, sel, :]

    # Feature extraction
    F_train = extract_features(Xp_train, cfg.features)
    F_val = extract_features(Xp_val, cfg.features)

    # Optional extra features
    if ns.add_time_features or ns.add_stft_features:
        Fe_extra_tr, names_tr = extract_extra_features(Xp_train, sfreq=cfg.preprocess.sfreq, use_time=ns.add_time_features, use_stft=ns.add_stft_features)
        Fe_extra_va, names_va = extract_extra_features(Xp_val, sfreq=cfg.preprocess.sfreq, use_time=ns.add_time_features, use_stft=ns.add_stft_features)
        # concat
        if Fe_extra_tr.size:
            F_train = np.hstack([F_train, Fe_extra_tr])
        if Fe_extra_va.size:
            F_val = np.hstack([F_val, Fe_extra_va])

    # Data balancing (train only)
    balance_info = {'strategy': 'none', 'train_counts_before': None, 'train_counts_after': None}
    balance_info['strategy'] = ns.balance if ns.balance else 'none'
    if F_train.size:
        uniq_b, cnts_b = np.unique(y_train, return_counts=True)
        balance_info['train_counts_before'] = dict(zip(uniq_b.tolist(), cnts_b.tolist()))
    if ns.balance and ns.balance != 'none':
        if RandomOverSampler is None or RandomUnderSampler is None:
            print('[balance] imbalanced-learn not available; skipping balancing')
        else:
            if ns.balance == 'undersample':
                rus = RandomUnderSampler(random_state=cfg.model.random_state)
                try:
                    F_train, y_train = rus.fit_resample(F_train, y_train)
                    print(f'[balance] undersample applied: new train counts={dict(zip(*np.unique(y_train, return_counts=True)))}')
                except Exception as e:
                    print('[balance] undersample failed:', e)
            elif ns.balance == 'oversample':
                ros = RandomOverSampler(random_state=cfg.model.random_state)
                try:
                    F_train, y_train = ros.fit_resample(F_train, y_train)
                    print(f'[balance] oversample applied: new train counts={dict(zip(*np.unique(y_train, return_counts=True)))}')
                except Exception as e:
                    print('[balance] oversample failed:', e)
    # record after counts
    if F_train.size:
        uniq_a, cnts_a = np.unique(y_train, return_counts=True)
        balance_info['train_counts_after'] = dict(zip(uniq_a.tolist(), cnts_a.tolist()))

    # Build models and evaluate
    models = build_models(cfg.model.random_state, cfg.model.class_weight)
    os.makedirs(ns.export_dir, exist_ok=True)
    results = {}
    for name, model in models.items():
        model.fit(F_train, y_train)
        # get probabilities or normalized decision function
        if hasattr(model[-1], 'predict_proba'):
            y_prob = model.predict_proba(F_val)[:, 1]
        else:
            s = model.decision_function(F_val)
            s = s.ravel() if s.ndim > 1 else s
            s_min, s_max = float(np.min(s)), float(np.max(s))
            y_prob = (s - s_min) / (s_max - s_min + 1e-8)

        # postprocessing: moving average smoothing on probabilities
        if ns.post_ma and ns.post_ma > 1:
            y_prob = pd.Series(y_prob).rolling(ns.post_ma, min_periods=1, center=False).mean().to_numpy()

        # binary predictions by threshold
        y_pred = (y_prob >= ns.threshold).astype(int)

        # k-of-n voting: sliding window decision (requires post_kofn as 'K,N')
        if ns.post_kofn:
            try:
                K, N = map(int, ns.post_kofn.split(','))
                from collections import deque
                win = deque(maxlen=N)
                y_pred_kofn = np.zeros_like(y_pred)
                for i, v in enumerate(y_pred):
                    win.append(v)
                    if len(win) == N and sum(win) >= K:
                        y_pred_kofn[i] = 1
                y_pred = y_pred_kofn
            except Exception:
                pass

        m = compute_metrics(y_val, y_prob, y_pred, beta=ns.beta)
        results[name] = m
        dump(model, os.path.join(ns.export_dir, f'model_{name}.joblib'))
        np.save(os.path.join(ns.export_dir, f'{name}_y_prob.npy'), y_prob)
        np.save(os.path.join(ns.export_dir, f'{name}_y_pred.npy'), y_pred)
        print(f'[model {name}]', m)

    # Persist metrics (JSON + CSV) and selected channels
    summary = {
        'meta': {
            'threshold': ns.threshold,
            'beta': ns.beta,
            'min_consecutive': ns.min_consecutive,
            't_on': ns.t_on if ns.t_on is not None else ns.threshold,
            't_off': ns.t_off if ns.t_off is not None else ns.threshold,
            'min_consecutive_on': ns.min_consecutive_on if ns.min_consecutive_on is not None else ns.min_consecutive,
            'min_consecutive_off': ns.min_consecutive_off,
            'refractory': ns.refractory,
            'max_channels': ns.max_channels,
            'n_train_epochs': int(X_train.shape[0]),
            'n_eval_epochs': int(X_val.shape[0]),
            'selected_channels': list(map(int, sel)),
            'selected_channel_names': sel_names,
            'balance': balance_info,
        },
        'results': results,
    }
    with open(os.path.join(ns.export_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    # CSV: one row per model
    rows = []
    for name, met in results.items():
        row = {'model': name}
        row.update({k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in met.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(ns.export_dir, 'metrics.csv'), index=False)
    with open(os.path.join(ns.export_dir, 'selected_channels.json'), 'w') as f:
        json.dump({'selected_indices': list(map(int, sel)), 'selected_names': sel_names, 'all_channel_names': all_names}, f, indent=2)

    # Simple alarm traces from best model (by ROC-AUC)
    best = max(results.items(), key=lambda kv: kv[1]['roc_auc'] if kv[1]['roc_auc']==kv[1]['roc_auc'] else -1)[0]
    print('[best]', best)
    best_prob = np.load(os.path.join(ns.export_dir, f'{best}_y_prob.npy'))
    alarms = alarm_from_probs(
        best_prob,
        threshold=ns.threshold,
        min_consecutive=ns.min_consecutive,
        t_on=ns.t_on,
        t_off=ns.t_off,
        min_consecutive_on=ns.min_consecutive_on,
        min_consecutive_off=ns.min_consecutive_off,
        refractory=ns.refractory,
    )
    np.save(os.path.join(ns.export_dir, f'{best}_alarms.npy'), alarms)
    print('[alarm] triggered epochs:', int(alarms.sum()))

if __name__ == '__main__':
    main()
