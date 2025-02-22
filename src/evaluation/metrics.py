import numpy as np
import typing 
import torch
from sklearn.metrics import roc_auc_score
from typing import List


def compute_cir_scores(predictions: np.ndarray, labels: np.ndarray):
    accuracy = np.mean(predictions == labels)
    return {
        'acc': float(accuracy)
    }

    
def compute_cp_scores(predictions: np.ndarray, labels: np.ndarray):
    predictions = (predictions > 0.5).astype(int)
    
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = np.mean(predictions == labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    auc = roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    
    return {
        'acc': float(accuracy), 
        'precision': float(precision), 
        'recall': float(recall), 
        'f1': float(f1),
        'auc': float(auc)
    }
