import json
import logging
import os
import pathlib
import pickle
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.distributed import cleanup, setup
from ..utils.logger import get_logger
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = "{polyvore_dir}/precomputed_clip_embeddings"
    

def collate_fn(batch):
    return [item for item in batch]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=128)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):
    item_dataset = polyvore.PolyvoreItemDataset(
        dataset_dir=args.polyvore_dir, load_image=True
    )

    n_items = len(item_dataset)
    n_items_per_gpu = n_items // world_size

    start_idx = n_items_per_gpu * rank
    end_idx = (start_idx + n_items_per_gpu) if rank < world_size - 1 else n_items
    item_dataset = torch.utils.data.Subset(item_dataset, range(start_idx, end_idx))
    
    item_dataloader = DataLoader(
        dataset=item_dataset, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn
    )

    return item_dataloader


def compute(rank: int, world_size: int, args: Any):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    logger = get_logger('precompute_clip_embedding', LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    item_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint).to(rank)
    model.eval()
    logger.info(f'Model Loaded')
    
    all_ids, all_embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(item_dataloader):
            if args.demo and len(all_embeddings) > 10:
                break
            
            embeddings = model.precompute_clip_embedding(batch)  # (batch_size, d_embed)
            
            all_ids.extend([item.item_id for item in batch])
            all_embeddings.append(embeddings)
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Computed {len(all_embeddings)} embeddings")

    # numpy 어레이 저장
    save_dir = POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/polyvore_{rank}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)
    
    # DDP 종료
    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    mp.spawn(
        compute, args=(args.world_size, args), 
        nprocs=args.world_size, join=True
    )