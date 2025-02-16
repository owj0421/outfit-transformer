import numpy as np
import typing 
import torch
from sklearn.metrics import roc_auc_score
from typing import List


def compute_cir_scores(
    predictions: np.array,
    labels: np.array
):
    return {
        'acc': (predictions == labels).mean()
    }


def compute_cp_scores(
    predictions: np.array,
    labels: np.array
):
    predictions = (predictions > 0.5).astype(int)
    
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    
    accuracy = (predictions == labels).mean()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (labels == 1).sum() if (labels == 1).sum() > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    if len(np.unique(labels)) == 1:
        auc = 0.0
    else:
        auc = roc_auc_score(y_true=labels,y_score=predictions)
    
    return {
        'acc': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'auc': auc
    }