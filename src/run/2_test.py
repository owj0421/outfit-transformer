import os
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from ..model.load import (
    load_model
)
from ..utils.loss import (
    focal_loss,
)
from ..utils.utils import (
    seed_everything,
)

import sys

from fashion_recommenders import datatypes
from fashion_recommenders.datasets import polyvore
from fashion_recommenders.stores.metadata import ItemMetadataStore

# from fashion_recommenders.utils.metrics import score_cp, score_fitb
from fashion_recommenders.metrics.compatibility import CompatibilityMetricCalculator
from fashion_recommenders.metrics.complementary import ComplementaryMetricCalculator

import pathlib


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
LOADER_DIR = SRC_DIR / 'stores'
RESULT_DIR = SRC_DIR / 'results'

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type', 
        type=str, required=True, choices=['original', 'clip'], default='original'
    )
    parser.add_argument(
        '--polyvore_dir', 
        type=str, required=True
    )
    parser.add_argument(
        '--polyvore_type', 
        type=str, required=True, choices=['nondisjoint', 'disjoint']
    )
    parser.add_argument(
        '--task',
        type=str, required=True, choices=['cp', 'fitb', 'cir']
    )
    parser.add_argument(
        '--batch_sz', 
        type=int, default=32
    )
    parser.add_argument(
        '--n_workers', 
        type=int, default=4
    )
    parser.add_argument(
        '--seed', 
        type=int, default=42
    )
    parser.add_argument(
        '--checkpoint',
        type=str, default=None
    )
    parser.add_argument(
        '--demo',
        action='store_true'
    )
    return parser.parse_args()


def cp_valid(args):
    metric = CompatibilityMetricCalculator()

    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    test = polyvore.PolyvoreCompatibilityDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='test'
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=test.collate_fn
    )
    
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    model.eval()
    
    pbar = tqdm(
        test_dataloader,
        desc=f'[Test] Compatibility'
    )
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
            predictions = model.predict(queries=data['query']).squeeze()
            score = metric.add(
                predictions=predictions.detach().cpu().numpy(),
                labels=np.array(data['label'])
            )
            pbar.set_postfix(**score)
            
    score = metric.calculate()
    print(
        f"[Test] Compatibility --> {score}"
    )
    
    result_dir = os.path.join(
        RESULT_DIR, args.checkpoint.split('/')[-2],
    )
    os.makedirs(
        result_dir, exist_ok=True
    )
    with open(os.path.join(result_dir, f'{args.task}_results.json'), 'w') as f:
        json.dump(score, f)
    print(
        f"[Test] Compatibility --> Results saved to {result_dir}"
    )


def fitb_valid(args):
    metric = ComplementaryMetricCalculator()
    
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    test = polyvore.PolyvoreFillInTheBlankDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='test'
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=test.collate_fn
    )
    
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    model.eval()
    
    pbar = tqdm(
        test_dataloader,
        desc=f'[Test] Fill in the Blank'
    )
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break

            queries = data['query']
            query_embedddings = model.embed_query(
                queries=queries
            )
            
            answers = data['answers']
            answer_embedddings = model.embed_item(
                items=sum(answers, [])
            ).view(args.batch_sz, 4, -1).detach().cpu().numpy()
            answer_embedddings = [answer_embedddings[i] for i in range(args.batch_sz)]
            
            
            score = metric.add(
                query_embeddings=[q.detach().cpu().numpy() for q in query_embedddings],
                candidate_embeddings=answer_embedddings,
                labels=np.array(data['label'])
            )
            pbar.set_postfix(**score)
            
    score = metric.calculate()
    print(
        f"[Test] Fill in the Blank --> {score}"
    )
    
    result_dir = os.path.join(
        RESULT_DIR, args.checkpoint.split('/')[-2],
    )
    os.makedirs(
        result_dir, exist_ok=True
    )
    with open(os.path.join(result_dir, f'{args.task}_results.json'), 'w') as f:
        json.dump(score, f)
    print(
        f"[Test] Fill in the Blank --> Results saved to {result_dir}"
    )

def cir_valid():
    pass


if __name__ == '__main__':
    args = parse_args()
    seed_everything(
        args.seed
    )
    if args.task == 'fitb':
        args.task = 'cir'
    
    if args.task == 'cp':
        cp_valid(args)
    elif args.task == 'cir':
        fitb_valid(args)