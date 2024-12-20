import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score_fitb(
    query_embed: torch.Tensor, # (batch_size, embedding_dim)
    candidate_embeds: torch.Tensor, # (batch_size, num_candidates, embedding_dim)
    label: torch.Tensor, # (batch_size)
):
    batch_sz, num_candidates, d_embed = candidate_embeds.size()
    
    query_cand_dist = torch.nn.functional.pairwise_distance(
        query_embed.unsqueeze(1).repeat(1, num_candidates, 1).reshape(-1, d_embed),
        candidate_embeds.reshape(-1, d_embed)
    )
    dist = query_cand_dist.view(batch_sz, num_candidates)
    pred = dist.min(dim=1).indices
    acc = (pred == label).sum().item() / len(label)
    
    return pred, {"acc": acc}


def score_cp(
    pred: torch.Tensor, # (batch_size, num_candidates)
    label: torch.Tensor, # (batch_size)
    compute_auc: bool = False
):
    acc = (
        (pred > 0.5).eq(label).sum().item() / len(label)
    )
    if not compute_auc:
        return {"acc": acc}
    
    try: 
        auc = roc_auc_score(
            y_true=pred.numpy(),
            y_score=label.numpy()
        )
    except ValueError:
        auc = 0.0
        
    return {"acc": acc, "auc": auc}