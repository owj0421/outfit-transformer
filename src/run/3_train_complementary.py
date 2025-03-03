import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Optional

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
from ..evaluation.metrics import compute_cir_scores, compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import InBatchTripletMarginLoss
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

metadata = None
all_embeddings_dict = None

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
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):    
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='valid', metadata=metadata, embedding_dict=embedding_dict
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn, sampler=valid_sampler
        )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    model.train()
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        batched_q_emb = model(data['query'], use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        batched_a_emb = model(data['answer'], use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        
        loss = loss_fn(batched_q_emb, batched_a_emb)
        loss = loss / args.accumulation_steps
        
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)  # (batch_sz, batch_sz)
        preds = torch.argmin(dists, dim=1) # (batch_sz,)
        labels = torch.arange(len(preds), device=rank)

        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'train_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output


@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        batched_q_emb = model(data['query'], use_precomputed_embedding=True).unsqueeze(1) # (batch_sz, 1, embedding_dim)
        batched_c_embs = model(sum(data['candidates'], []), use_precomputed_embedding=True) # (batch_sz * 4, embedding_dim)
        batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1]) # (batch_sz, 4, embedding_dim)
        
        dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1) # (batch_sz, 4)
        preds = torch.argmin(dists, dim=-1) # (batch_sz,)
        labels = torch.tensor(data['label'], device=rank)

        # Accumulate Results
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        logs = {
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    _, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {**compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'valid_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output

    
def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'complementary_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = InBatchTripletMarginLoss(margin=2.0, reduction='mean')
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # Training Loop
    for epoch in range(args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, optimizer, scheduler, loss_fn, train_dataloader
        )

        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, loss_fn, valid_dataloader
        )
        
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict()
            }, checkpoint_path)
            
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
            
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cir', config=args.__dict__)
    else:
        wandb_run = None
    
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    )