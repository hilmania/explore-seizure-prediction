import numpy as np
from typing import List, Tuple
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer

# Wrapper-based sequential forward selection over channels

def sfs_channels(X: np.ndarray, y: np.ndarray, clf, max_channels: int = 8, scoring: str = 'roc_auc', cv_folds: int = 3, step: int = 1, random_state: int = 42) -> List[int]:
    # X: (n_samples, n_channels, n_samples_per_epoch) -> we will featureize inside scoring by mean power to speed selection
    # However better approach: collapse samples by simple stats for selection stage
    rng = np.random.RandomState(random_state)
    n_ch = X.shape[1]
    remaining = list(range(n_ch))
    selected: List[int] = []
    scorer = get_scorer(scoring)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def summarize(epoch):
        # simple summary per channel to use during selection (RMS + band powers rough)
        from scipy.signal import welch
        ftrs = []
        for c in range(epoch.shape[0]):
            x = epoch[c]
            rms = np.sqrt(np.mean(x**2))
            ftrs.append(rms)
            f, Pxx = welch(x, fs=256.0, nperseg=min(256, len(x)))
            def bp(lo, hi):
                m = (f >= lo) & (f < hi)
                return np.trapz(Pxx[m], f[m])
            ftrs.extend([bp(0.5,4), bp(4,8), bp(8,13), bp(13,30)])
        return np.array(ftrs)

    # precompute summary features for speed
    Xs = np.array([summarize(ep) for ep in X])
    # map from channel indices to column spans
    cols_per_ch = 5  # rms + 4 bands

    while len(selected) < max_channels and remaining:
        scores: List[Tuple[int, float]] = []
        for ch in remaining:
            cols = []
            for sc in selected + [ch]:
                cols.extend(list(range(sc*cols_per_ch, sc*cols_per_ch+cols_per_ch)))
            Xsub = Xs[:, cols]
            model = clone(clf)
            s = np.mean(cross_val_score(model, Xsub, y, scoring=scoring, cv=cv, n_jobs=None))
            scores.append((ch, s))
        scores.sort(key=lambda t: t[1], reverse=True)
        best = [ch for ch,_ in scores[:step]]
        for b in best:
            selected.append(b)
            remaining.remove(b)
    return selected
