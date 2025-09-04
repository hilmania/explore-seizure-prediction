from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> Dict[str, float]:
    # y_prob: probability for class 1
    # Specificity = TN / (TN + FP)
    # Sensitivity = Recall = TP / (TP + FN)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # F-beta
    if beta <= 0:
        beta = 1.0
    beta2 = beta*beta
    try:
        fb = (1+beta2) * (prec*rec) / (beta2*prec + rec)
    except ZeroDivisionError:
        fb = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn+fp)>0 else 0.0
    fpr = fp / (fp + tn) if (fp+tn)>0 else 0.0
    return {
        'roc_auc': auc,
        'accuracy': acc,
        'precision': prec,
        'sensitivity': rec,
        'specificity': spec,
        'f1': f1,
        'f_beta': fb,
        'fpr': fpr,
        'tp': float(tp), 'tn': float(tn), 'fp': float(fp), 'fn': float(fn)
    }
