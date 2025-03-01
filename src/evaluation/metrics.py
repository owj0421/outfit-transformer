import numpy as np
import typing 
import torch
from sklearn.metrics import roc_auc_score
from typing import List


def compute_cir_scores(predictions: torch.Tensor, labels: torch.Tensor):
    accuracy = torch.mean((predictions == labels).float()).item()
    return {
        'acc': accuracy
    }

def compute_cp_scores(predictions: torch.Tensor, labels: torch.Tensor):
    auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy()) if len(torch.unique(labels)) > 1 else 0.0
    auc = float(auc)
    
    predictions = (predictions > 0.5).int()
    
    tp = torch.sum((predictions == 1) & (labels == 1)).item()
    fp = torch.sum((predictions == 1) & (labels == 0)).item()
    fn = torch.sum((predictions == 0) & (labels == 1)).item()
    
    accuracy = torch.mean((predictions == labels).float()).item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'acc': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'auc': auc
    }

