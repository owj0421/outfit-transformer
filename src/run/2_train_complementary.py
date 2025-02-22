import os
import json
import numpy as np
import logging
import torch
import pathlib
import wandb
from typing import Any, Optional
from tqdm import tqdm
from argparse import ArgumentParser
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from ..models.load import load_model
from ..utils.loss import focal_loss
from ..utils.utils import seed_everything
from ..utils.logger import get_logger
from ..utils.distributed import setup, cleanup
from ..data.datasets import polyvore
from ..data import collate_fn
from ..evaluation.metrics import compute_cp_scores, compute_cir_scores


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
                        default=48)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=8)
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
    global metadata, all_embeddings_dict
    
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, all_embeddings_dict=all_embeddings_dict
    )
    valid = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='valid', metadata=metadata, all_embeddings_dict=all_embeddings_dict
    )

    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    valid_sampler = DistributedSampler(
        valid, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    train_dataloader = DataLoader(
        dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=train_sampler
    )
    valid_dataloader = DataLoader(
        dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=valid_sampler
    )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    model.train()
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}')
    
    predictions, labels = [], []
    total_loss = torch.zeros(1, device=rank)  # Initialize total loss as a tensor on the current device
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        batched_q_emb = model(data['query'], use_precomputed_embedding=True)
        batched_a_emb = model(data['answer'], use_precomputed_embedding=True)
        
        n_candidates = args.batch_sz_per_gpu # 1 positive + n-1 negative
            
        batched_c_embs = []
        for i in range(args.batch_sz_per_gpu):
            batched_n_embs = [batched_a_emb[j] for j in range(n_candidates) if i != j]
            batched_c_embs.append(torch.stack([batched_a_emb[i]] + batched_n_embs))
        
        loss = 0
        for b_i, (q_emb, c_embs) in enumerate(zip(batched_q_emb, batched_c_embs)):
            loss += loss_fn(
                q_emb.unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                c_embs[0, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                c_embs[1:, :] # (n_candidates-1, embedding_dim)
            )
        loss = loss / (args.accumulation_steps * (b_i+1))
        
        dists = torch.stack([
            torch.sum(
                torch.norm(q_emb.detach()[None, None, :] - c_embs.detach()[None, :, :], dim=2), dim=0
            )
            for q_emb, c_embs in zip(batched_q_emb, batched_c_embs)
        ])  # Shape: (batch_sz, num_candidates)

        predictions_ = torch.argmin(dists, dim=1)  # Shape: (batch_sz,)
        predictions.append(predictions_)  # Append as torch.Tensor
        labels.append(torch.zeros(args.batch_sz_per_gpu, dtype=torch.long, device=rank))  # Directly on GPU
        
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * args.accumulation_steps  # Accumulate scaled loss

        # Logging
        score = compute_cir_scores(predictions[-1].cpu().numpy(), labels[-1].cpu().numpy())
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
    
    predictions = torch.cat(predictions).to(rank)
    labels = torch.cat(labels).to(rank)

    gathered_predictions = [torch.empty_like(predictions) for _ in range(world_size)]
    gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(gathered_predictions, predictions)
    dist.all_gather(gathered_labels, labels)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= (len(dataloader) * world_size)

    if rank == 0:
        all_predictions = torch.cat(gathered_predictions).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        all_score = compute_cp_scores(all_predictions, all_labels)
        logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End (Score: {all_score}, Avg Loss: {total_loss.item()})')
    else:
        all_score = {}
        logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End')

    return {
        'train_loss': total_loss.item(), 
        **{f'train_{key}': value for key, value in all_score.items()}
    }


        
        
@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}')
    
    predictions, labels = [], []
    total_loss = torch.zeros(1, device=rank)
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        batched_q_emb = model(data['query'], use_precomputed_embedding=True)
        batched_a_emb = model(data['answer'], use_precomputed_embedding=True)
        
        n_candidates = args.batch_sz_per_gpu # 1 positive + n-1 negative
            
        batched_c_embs = []
        for i in range(args.batch_sz_per_gpu):
            batched_n_embs = [batched_a_emb[j] for j in range(n_candidates) if i != j]
            batched_c_embs.append(torch.stack([batched_a_emb[i]] + batched_n_embs))
        
        loss = 0
        for b_i, (q_emb, c_embs) in enumerate(zip(batched_q_emb, batched_c_embs)):
            loss += loss_fn(
                q_emb.unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                c_embs[0, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                c_embs[1:, :] # (n_candidates-1, embedding_dim)
            )
        loss = loss / (args.accumulation_steps * (b_i+1))
        
        dists = torch.stack([
            torch.sum(
                torch.norm(q_emb.detach()[None, None, :] - c_embs.detach()[None, :, :], dim=2), dim=0
            )
            for q_emb, c_embs in zip(batched_q_emb, batched_c_embs)
        ])  # Shape: (batch_sz, num_candidates)

        predictions_ = torch.argmin(dists, dim=1)  # Shape: (batch_sz,)
        predictions.append(predictions_)  # Append as torch.Tensor
        labels.append(torch.zeros(args.batch_sz_per_gpu, dtype=torch.long, device=rank))  # Directly on GPU
        
        total_loss += loss.item() * args.accumulation_steps  # Accumulate scaled loss

        # Logging
        score = compute_cir_scores(predictions[-1].cpu().numpy(), labels[-1].cpu().numpy())
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    predictions = torch.cat(predictions).to(rank)
    labels = torch.cat(labels).to(rank)

    gathered_predictions = [torch.empty_like(predictions) for _ in range(world_size)]
    gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(gathered_predictions, predictions)
    dist.all_gather(gathered_labels, labels)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= (len(dataloader) * world_size)

    if rank == 0:
        all_predictions = torch.cat(gathered_predictions).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        all_score = compute_cp_scores(all_predictions, all_labels)
        logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End (Score: {all_score}, Avg Loss: {total_loss.item()})')
    else:
        all_score = {}
        logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End')

    return {
        'valid_loss': total_loss.item(), 
        **{f'valid_{key}': value for key, value in all_score.items()}
    }

    
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
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=len(train_dataloader),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = torch.nn.TripletMarginLoss(margin=2.0, p=2)
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # Training Loop
    for epoch in range(args.n_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            ddp_model, optimizer, scheduler, loss_fn, train_dataloader
        )
        
        valid_dataloader.sampler.set_epoch(epoch)
        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            ddp_model, loss_fn, valid_dataloader
        )
        
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            
        if rank == 0:
            torch.save({
                'config': ddp_model.module.cfg.__dict__,
                'model': ddp_model.state_dict()
            }, checkpoint_path)
            
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
            
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        ddp_model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer', config=args.__dict__)
    else:
        wandb_run = None
        
    metadata = polyvore.load_metadata(args.polyvore_dir)
    all_embeddings_dict = polyvore.load_all_embeddings_dict(args.polyvore_dir, metadata)
    
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    )