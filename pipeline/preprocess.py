from typing import Optional, Tuple
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter
from mne.preprocessing import ICA
from dataclasses import asdict
from .config import PreprocessConfig


def bandpass_filter(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    # data: (n_channels, n_samples)
    nyq = 0.5 * sfreq
    low = max(l_freq / nyq, 0.0001)
    high = min(h_freq / nyq, 0.9999)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def notch_filter(data: np.ndarray, sfreq: float, freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
    if freq is None:
        return data
    w0 = freq / (sfreq / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data, axis=-1)


def run_ica_epoch(epoch: np.ndarray, sfreq: float, n_components: int, max_iter: int) -> np.ndarray:
    # epoch: (n_channels, n_samples)
    # Use MNE ICA on raw-like array
    # We fake an Info with arbitrary channel names
    import mne
    chs = [f'EEG{i:02d}' for i in range(epoch.shape[0])]
    info = mne.create_info(chs, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(epoch, info, verbose='ERROR')
    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=97, verbose='ERROR')
    ica.fit(raw)
    # Here we don't auto-detect EOG; instead remove components with kurtosis > threshold as a simple heuristic
    from scipy.stats import kurtosis
    sources = ica.get_sources(raw).get_data()
    kurt = kurtosis(sources, axis=1, fisher=True, bias=False)
    to_exclude = list(np.where(np.abs(kurt) > 5)[0])
    ica.exclude = to_exclude
    clean = ica.apply(raw.copy(), verbose='ERROR').get_data()
    return clean


def preprocess_epochs(X: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # X: (n_epochs, n_channels, n_samples)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        ep = X[i]
        ep = bandpass_filter(ep, cfg.sfreq, cfg.l_freq, cfg.h_freq)
        if cfg.notch:
            ep = notch_filter(ep, cfg.sfreq, cfg.notch)
        if cfg.do_ica and cfg.ica_n_components and cfg.ica_n_components > 0:
            try:
                ep = run_ica_epoch(ep, cfg.sfreq, cfg.ica_n_components, cfg.ica_max_iter)
            except Exception:
                # fallback without ICA
                pass
        out[i] = ep
    return out
