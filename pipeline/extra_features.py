import numpy as np
from typing import Dict
from scipy.signal import stft
from scipy.stats import skew, kurtosis

def time_domain_stats_epoch(epoch: np.ndarray) -> Dict[str, float]:
    # epoch: (n_channels, n_samples)
    feats = {}
    for c in range(epoch.shape[0]):
        x = epoch[c]
        feats[f'ch{c}_mean'] = float(np.mean(x))
        feats[f'ch{c}_std'] = float(np.std(x))
        feats[f'ch{c}_skew'] = float(skew(x))
        feats[f'ch{c}_kurt'] = float(kurtosis(x))
        feats[f'ch{c}_ptp'] = float(np.ptp(x))
        feats[f'ch{c}_rms'] = float(np.sqrt(np.mean(x**2)))
    return feats

def stft_bandpower_epoch(epoch: np.ndarray, sfreq: float = 256.0, bands=None) -> Dict[str, float]:
    # compute STFT and band power per channel (averaged over time)
    if bands is None:
        bands = [(0.5,4,'delta'), (4,8,'theta'), (8,13,'alpha'), (13,30,'beta')]
    feats = {}
    for c in range(epoch.shape[0]):
        x = epoch[c]
        f, t, Z = stft(x, fs=sfreq, nperseg=min(256, len(x)))
        P = np.abs(Z)**2
        for lo, hi, name in bands:
            mask = (f >= lo) & (f < hi)
            if np.any(mask):
                bp = np.mean(np.trapz(P[mask,:], f[mask], axis=0))
            else:
                bp = 0.0
            feats[f'ch{c}_stft_bp_{name}'] = float(bp)
    return feats

def extract_extra_features(X: np.ndarray, sfreq: float = 256.0, use_time=True, use_stft=True) -> (np.ndarray, list):
    # returns feature matrix and list of feature names
    feat_list = []
    for i in range(X.shape[0]):
        feats = {}
        if use_time:
            feats.update(time_domain_stats_epoch(X[i]))
        if use_stft:
            feats.update(stft_bandpower_epoch(X[i], sfreq=sfreq))
        # sort keys for deterministic order
        feat_list.append([feats[k] for k in sorted(feats.keys())])
    feat_names = sorted(feats.keys()) if feat_list else []
    return np.asarray(feat_list, dtype=float), feat_names
