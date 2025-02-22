import os
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from ..models.load import load_model
from ..utils.loss import focal_loss
from ..utils.utils import seed_everything
from ..data.datasets import polyvore
from ..data import collate_fn
from ..evaluation.metrics import compute_cp_scores
import pathlib


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
                        default=64)
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
    all_embeddings_dict = polyvore.load_all_embeddings_dict(args.polyvore_dir)
    
    test = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='test', metadata=metadata, all_embeddings_dict=all_embeddings_dict
    )
    test_dataloader = DataLoader(
        dataset=test, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
    )
    
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint) 
    model = model.cuda()
    model.eval()
    
    pbar = tqdm(test_dataloader, desc=f'[Test] Compatibility')
    predictions, labels = [], []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 2:
                break
            labels_ = torch.tensor(data['label'], dtype=torch.float32, device='cuda')
            predictions_ = model(data['query'], use_precomputed_embedding=True).squeeze(1)
            
            predictions.append(predictions_.detach().cpu().numpy())
            labels.append(labels_.detach().cpu().numpy())
        
            score = compute_cp_scores(predictions[-1], labels[-1])
            pbar.set_postfix(**score)
    
    score = compute_cp_scores(np.concatenate(predictions), np.concatenate(labels))
    print(
        f"[Test] Compatibility --> {score}"
    )
    
    if args.checkpoint:
        result_dir = os.path.join(
            RESULT_DIR, args.checkpoint.split('/')[-2],
        )
    else:
        result_dir = os.path.join(
            RESULT_DIR, 'compatibility_demo',
        )
    os.makedirs(
        result_dir, exist_ok=True
    )
    with open(os.path.join(result_dir, f'results.json'), 'w') as f:
        json.dump(score, f)
    print(
        f"[Test] Compatibility --> Results saved to {result_dir}"
    )


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    validation(args)