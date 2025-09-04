import os
import glob
import numpy as np
from typing import Tuple, List

# Assumes X.npy shape: (n_epochs, n_channels, n_samples) and y.npy shape: (n_epochs,)
# Labels: 1 preictal, 0 interictal/normal

def list_Xy(split_dir: str) -> List[Tuple[str, str]]:
    Xs = sorted(glob.glob(os.path.join(split_dir, '*', '*', '*', '*_X.npy')))
    pairs = []
    for x in Xs:
        y = x.replace('_X.npy', '_y.npy')
        if os.path.exists(y):
            pairs.append((x, y))
    return pairs


def load_split(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    pairs = list_Xy(os.path.join(root, split))
    Xs, ys = [], []
    for x_path, y_path in pairs:
        X = np.load(x_path)
        y = np.load(y_path)
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise FileNotFoundError(f'No data found under {root}/{split}')
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def describe(X: np.ndarray, y: np.ndarray) -> str:
    uniq, cnts = np.unique(y, return_counts=True)
    return f'shape={X.shape}, y_counts={dict(zip(uniq.tolist(), cnts.tolist()))}'
