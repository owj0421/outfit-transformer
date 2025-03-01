# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""

import os
import math
import wandb
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class InBatchTripletMarginLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self, batched_q_emb: Tensor, batched_a_emb: Tensor
    ):
        batch_size = batched_q_emb.shape[0]
        # Compute pairwise distance matrix
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)  # (batch_size, batch_size)
        # Positive distances (diagonal elements: query-answer pairs)
        pos_dists = torch.diag(dists)  # (batch_size,)
        # Negative distances (all other pairs)
        neg_dists = dists.clone()  # Copy distance matrix
        neg_dists.fill_diagonal_(float('inf'))  # Ignore diagonal (positive pairs)
        hardest_neg_dists, _ = neg_dists.min(dim=1)  # Select the hardest negative for each query
        # Compute triplet loss
        loss = F.relu(pos_dists - hardest_neg_dists + self.margin)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):  
        super().__init__()
        assert gamma >= 0, (
            f"Invalid Value for arg 'gamma': '{gamma}' \n Gamma should be non-negative"
        )
        assert 0 <= alpha <= 1, (
            f"Invalid Value for arg 'alpha': '{alpha}' \n Alpha should be in range [0, 1]"
        )
        assert reduction in ['none', 'mean', 'sum'], (
            f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, y_prob: torch.Tensor, y_true: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(y_prob, y_true, reduction="none")
        p_t = y_prob * y_true + (1 - y_prob) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / torch.tensor(loss.numel(), device=loss.device)  # DDP safe  


def safe_divide(a, b, eps=1e-7):
    return a / (b + eps)
