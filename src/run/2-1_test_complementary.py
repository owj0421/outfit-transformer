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
from ..evaluation.metrics import compute_cir_scores
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
                        default='./polyvore')
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
    test = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='test'
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=args.batch_sz_per_gpu,
        shuffle=True,
        num_workers=args.n_workers_per_gpu,
        collate_fn=test.collate_fn
    )
    
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    model.eval()
    
    pbar = tqdm(test_dataloader, desc=f'[Test] Fill in the Blank')
    predictions, labels = [], []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 2:
                break

            batched_q = data['query']
            batched_cs = data['answers']
        
            batched_q_emb = model.embed_complementary_query(
                query=batched_q
            ) # List(batch_sz) of Tensor(embed_sz)

            batched_c_embs = model.embed_complementary_item(
                item=sum(batched_cs, [])
            ) # List(batch_sz * 4) of Tensor(embed_sz)
            batched_c_embs = torch.stack(batched_c_embs).view(args.batch_sz_per_gpu, 4, -1) # (batch_sz, 4, embed_sz)
            
            dists = np.array([
                np.sum(
                    np.linalg.norm(q_emb.detach().cpu().numpy()[None, None, :] - c_embs.detach().cpu().numpy()[None, :, :], axis=2), axis=0
                )
                for q_emb, c_embs in zip(batched_q_emb, batched_c_embs)
            ])  # Shape: (batch_sz, num_candidates)
            predictions_ = dists.argmin(axis=1)
            
            predictions.append(predictions_)
            labels.append(np.array(data['label']))
            
            score = compute_cir_scores(predictions[-1], labels[-1])
            
            pbar.set_postfix(**score)
            
    score = compute_cir_scores(np.concatenate(predictions), np.concatenate(labels))
    print(
        f"[Test] Fill in the Blank --> {score}"
    )
    
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
    print(
        f"[Test] Compatibility --> Results saved to {result_dir}"
    )



if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    validation(args)