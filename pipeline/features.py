import numpy as np
from typing import Dict
import antropy as ant
from dataclasses import asdict
from .config import FeatureConfig


def _safe(func, *a, **k):
    try:
        return func(*a, **k)
    except Exception:
        return np.nan


def chaos_features_epoch(epoch: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    # epoch: (n_channels, n_samples)
    feats = {}
    r_abs = cfg.sample_entropy_r
    for c in range(epoch.shape[0]):
        x = epoch[c]
        std = np.std(x) or 1.0
        r = r_abs if r_abs > 0 and r_abs < 1 else 0.2
        r *= std
        feats[f'ch{c}_sampen'] = _safe(ant.sample_entropy, x, order=cfg.sample_entropy_m, r=r)
        feats[f'ch{c}_perm_ent'] = _safe(ant.perm_entropy, x, order=cfg.permutation_entropy_order, delay=cfg.permutation_entropy_delay, normalize=True)
        feats[f'ch{c}_svd_ent'] = _safe(ant.svd_entropy, x, order=cfg.permutation_entropy_order, delay=cfg.permutation_entropy_delay, normalize=True)
        feats[f'ch{c}_higuchi_fd'] = _safe(ant.higuchi_fd, x)
        feats[f'ch{c}_petrosian_fd'] = _safe(ant.petrosian_fd, x)
        # chaos proxies
        dfa_fn = getattr(ant, 'detrended_fluctuation', None)
        feats[f'ch{c}_dfa'] = _safe(dfa_fn, x) if dfa_fn is not None else np.nan
        hurst_fn = getattr(ant, 'hurst_rs', None)
        if hurst_fn is None:
            hurst_fn = getattr(ant, 'hurst_exp', None)
        feats[f'ch{c}_hurst'] = _safe(hurst_fn, x) if hurst_fn is not None else np.nan
        # basic band powers could help stability
        try:
            from scipy.signal import welch
            f, Pxx = welch(x, fs=256.0, nperseg=min(256, len(x)))
            def bp(lo, hi):
                m = (f >= lo) & (f < hi)
                return np.trapz(Pxx[m], f[m])
            feats[f'ch{c}_bp_delta'] = bp(0.5, 4)
            feats[f'ch{c}_bp_theta'] = bp(4, 8)
            feats[f'ch{c}_bp_alpha'] = bp(8, 13)
            feats[f'ch{c}_bp_beta'] = bp(13, 30)
        except Exception:
            feats[f'ch{c}_bp_delta'] = np.nan
            feats[f'ch{c}_bp_theta'] = np.nan
            feats[f'ch{c}_bp_alpha'] = np.nan
            feats[f'ch{c}_bp_beta'] = np.nan
    return feats


def extract_features(X: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    # returns (n_epochs, n_features)
    feat_list = []
    for i in range(X.shape[0]):
        feats = chaos_features_epoch(X[i], cfg)
        feat_list.append([feats[k] for k in sorted(feats.keys())])
    return np.asarray(feat_list, dtype=float)
