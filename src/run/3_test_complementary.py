import json
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cir_scores
from ..models.load import load_model
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def validation(args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    test = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='test', metadata=metadata, embedding_dict=embedding_dict
    )
    test_dataloader = DataLoader(
        dataset=test, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
    )
    
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint) 
    model.eval()
    
    pbar = tqdm(test_dataloader, desc=f'[Test] Fill in the Blank')
    all_preds, all_labels = [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        batched_q_emb = model(data['query'], use_precomputed_embedding=True).unsqueeze(1) # (batch_sz, 1, embedding_dim)
        batched_c_embs = model(sum(data['candidates'], []), use_precomputed_embedding=True) # (batch_sz * 4, embedding_dim)
        batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1]) # (batch_sz, 4, embedding_dim)
        
        dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1) # (batch_sz, 4)
        preds = torch.argmin(dists, dim=-1) # (batch_sz,)
        labels = torch.tensor(data['label']).cuda()

        # Accumulate Results
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        logs = {
            **score
        }
        pbar.set_postfix(**logs)
    
    all_preds = torch.cat(all_preds).cuda()
    all_labels = torch.cat(all_labels).cuda()
    score = compute_cir_scores(all_preds, all_labels)
    print(f"[Test] Fill in the Blank --> {score}")
    
    if args.checkpoint:
        result_dir = os.path.join(
            RESULT_DIR, args.checkpoint.split('/')[-2],
        )
    else:
        result_dir = os.path.join(
            RESULT_DIR, 'complementary_demo',
        )
    os.makedirs(
        result_dir, exist_ok=True
    )
    with open(os.path.join(result_dir, f'results.json'), 'w') as f:
        json.dump(score, f)
    print(f"[Test] Fill in the Blank  --> Results saved to {result_dir}")



if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    validation(args)