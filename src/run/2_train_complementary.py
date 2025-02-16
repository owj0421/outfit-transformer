import os
import json
import numpy as np
import logging
import torch
import pathlib
import wandb
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
from ..utils.distributed import setup, cleanup
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cp_scores, compute_cir_scores


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_logger(name: str, log_dir: pathlib.Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_file = log_dir / f"{name}_log.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['original', 'clip'], default='original')
    parser.add_argument('--polyvore_dir', type=str, required=True)
    parser.add_argument('--polyvore_type', type=str, required=True, choices=['nondisjoint', 'disjoint'])
    parser.add_argument('--batch_sz_per_gpu', type=int, default=1)
    parser.add_argument('--n_workers_per_gpu', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='train'
    )
    valid = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='valid'
    )

    # Distributed sampler for training
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    valid_sampler = DistributedSampler(valid, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(
        dataset=train,
        batch_size=args.batch_sz_per_gpu,
        shuffle=False,  # Shuffling is handled by the sampler
        num_workers=args.n_workers_per_gpu,
        collate_fn=train.collate_fn,
        sampler=train_sampler
    )
    valid_dataloader = DataLoader(
        dataset=valid,
        batch_size=args.batch_sz_per_gpu,
        shuffle=False,
        num_workers=args.n_workers_per_gpu,
        collate_fn=valid.collate_fn,
        sampler=valid_sampler
    )

    return train_dataloader, valid_dataloader


# def save_checkpoint(model, optimizer, scheduler, epoch, score, loss, args):
#     save_dir = os.path.join(
#         CHECKPOINT_DIR, 
#         f"{args.model_type}-{wandb.run.name}",
#         f'epoch_{epoch+1}_acc_{score["acc"]:.3f}_auc_{score["auc"]:.3f}_loss_{loss:.3f}'
#     )
#     os.makedirs(save_dir, exist_ok=True)

#     with open(os.path.join(save_dir, 'args.json'), 'w') as f:
#         json.dump(args.__dict__, f)
    
#     torch.save(model.state_dict(), os.path.join(save_dir, f'model.pt'))
#     torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optimizer.pt'))
#     if scheduler:
#         torch.save(scheduler.state_dict(), os.path.join(save_dir, f'scheduler.pt'))


def train_step(
    rank, world_size, args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, train_dataloader
):
    model.train()
    
    pbar = tqdm(train_dataloader, desc=f'[GPU {rank}] Train Epoch {epoch+1}/{args.n_epochs}')
    
    predictions = []
    labels = []
    total_loss = torch.zeros(1, device=rank)  # Initialize total loss as a tensor on the current device

    for i, data in enumerate(pbar):

        if args.demo and i > 10:
            break
        
        batched_q = data['query']
        batched_a = data['answer']
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            batched_q_emb = model.module.embed_complementary_query(
                query=batched_q
            )
            batched_a_emb = model.module.embed_complementary_item(
                item=batched_a
            )
        else:
            batched_q_emb = model.embed_complementary_query(
                query=batched_q
            )
            batched_a_emb = model.embed_complementary_item(
                item=batched_a
            )
        
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
        
        dists = np.array([
            np.sum(
                np.linalg.norm(q_emb.detach().cpu().numpy()[None, None, :] - c_embs.detach().cpu().numpy()[None, :, :], axis=2), axis=0
            )
            for q_emb, c_embs in zip(batched_q_emb, batched_c_embs)
        ])  # Shape: (batch_sz, num_candidates)
        predictions_ = dists.argmin(axis=1)
        
        predictions.append(predictions_)
        labels.append(np.zeros(args.batch_sz_per_gpu))
        
        loss.backward()

        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * args.accumulation_steps  # Accumulate scaled loss

        score = compute_cir_scores(predictions[-1], labels[-1])
        log = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar),
            **score
        }
        log = {f'train_{key}': value for key, value in log.items() if not np.isnan(value)}

        if args.wandb_key:
            wandb_run.log(log)

        pbar.set_postfix(**log)
    
    # Synchronize all processes before gathering data
    dist.barrier()

    # Gather predictions and labels using all_gather_object
    local_data = {
        "predictions": np.concatenate(predictions).tolist(),  # Convert numpy array to list
        "labels": np.concatenate(labels).tolist()
    }

    # Collect data from all ranks
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)

    # Combine data from all ranks
    if rank == 0:
        all_predictions = []
        all_labels = []
        for data in gathered_data:
            all_predictions.extend(data["predictions"])
            all_labels.extend(data["labels"])
        
        # Convert back to numpy arrays for scoring
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute the final score
        score = compute_cir_scores(all_predictions, all_labels)
    else:
        score = None

    # Synchronize and reduce total_loss across all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    
    # Average the total_loss across world_size
    total_loss /= len(train_dataloader) * world_size

    if rank == 0:
        logger.info(f'[GPU {rank}] Train Epoch {epoch+1}/{args.n_epochs} --> End (Score: {score}, Avg Loss: {total_loss.item()})')
    else:
        logger.info(f'[GPU {rank}] Train Epoch {epoch+1}/{args.n_epochs} --> End')

    return score, total_loss.item()  # Return both score and average loss


        
        
@torch.no_grad()
def valid_step(
    rank, world_size, args, epoch, logger, wandb_run,
    model, loss_fn, valid_dataloader
):
    model.eval()
    
    pbar = tqdm(valid_dataloader, desc=f'[GPU {rank}] Valid Epoch {epoch+1}/{args.n_epochs}')
    
    predictions = []
    labels = []
    total_loss = torch.zeros(1, device=rank)  # Initialize total loss as a tensor on the current device
    
    for i, data in enumerate(pbar):

        if args.demo and i > 10:
            break
        
        batched_q = data['query']
        batched_a = data['answer']
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            batched_q_emb = model.module.embed_complementary_query(
                query=batched_q
            )
            batched_a_emb = model.module.embed_complementary_item(
                item=batched_a
            )
        else:
            batched_q_emb = model.embed_complementary_query(
                query=batched_q
            )
            batched_a_emb = model.embed_complementary_item(
                item=batched_a
            )
        
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
        
        dists = np.array([
            np.sum(
                np.linalg.norm(q_emb.detach().cpu().numpy()[None, None, :] - c_embs.detach().cpu().numpy()[None, :, :], axis=2), axis=0
            )
            for q_emb, c_embs in zip(batched_q_emb, batched_c_embs)
        ])  # Shape: (batch_sz, num_candidates)
        predictions_ = dists.argmin(axis=1)
        
        predictions.append(predictions_)
        labels.append(np.zeros(args.batch_sz_per_gpu))
        
        total_loss += loss.item() * args.accumulation_steps  # Accumulate scaled loss

        score = compute_cir_scores(predictions[-1], labels[-1])
        log = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar),
            **score
        }
        log = {f'valid_{key}': value for key, value in log.items() if not np.isnan(value)}

        if args.wandb_key:
            wandb_run.log(log)

        pbar.set_postfix(**log)
    
    # Synchronize all processes before gathering data
    dist.barrier()

    # Gather predictions and labels using all_gather_object
    local_data = {
        "predictions": np.concatenate(predictions).tolist(),  # Convert numpy array to list
        "labels": np.concatenate(labels).tolist()
    }

    # Collect data from all ranks
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)

    # Combine data from all ranks
    if rank == 0:
        all_predictions = []
        all_labels = []
        for data in gathered_data:
            all_predictions.extend(data["predictions"])
            all_labels.extend(data["labels"])
        
        # Convert back to numpy arrays for scoring
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute the final score
        score = compute_cir_scores(all_predictions, all_labels)
    else:
        score = None

    # Synchronize and reduce total_loss across all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    
    # Average the total_loss across world_size
    total_loss /= len(valid_dataloader) * world_size

    if rank == 0:
        logger.info(f'[GPU {rank}] Valid Epoch {epoch+1}/{args.n_epochs} --> End (Score: {score}, Avg Loss: {total_loss.item()})')
    else:
        logger.info(f'[GPU {rank}] Valid Epoch {epoch+1}/{args.n_epochs} --> End')

    return score, total_loss.item()  # Return both score and average loss


def train(rank, world_size, args, wandb_run):
    logger = get_logger(f"{wandb_run.name}_rank{rank}", LOGS_DIR)
    
    
    setup(rank, world_size)
    logger.info(f'[GPU {rank}] Setup DDP Completed')
    
    
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'[GPU {rank}] Dataloaders Setup Completed')
    
    
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint).to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # Wrap the model with DDP
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.n_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )
    loss_fn = torch.nn.TripletMarginLoss(margin=2.0, p=2)
    logger.info(f'[GPU {rank}] Model Loaded and Wrapped with DDP')


    for epoch in range(args.n_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        score, loss = train_step(
            rank, world_size, args, epoch, logger, wandb_run,
            ddp_model, optimizer, scheduler, loss_fn, train_dataloader
        )
        
        
        valid_dataloader.sampler.set_epoch(epoch)
        score, loss = valid_step(
            rank, world_size, args, epoch, logger, wandb_run,
            ddp_model, loss_fn, valid_dataloader
        )
        
        
        checkpoint_dir = os.path.join(
            CHECKPOINT_DIR, 
            f"complementary-{args.model_type}-{wandb.run.name}",
            f'epoch_{epoch+1}_acc_{score["acc"]:.3f}_auc_{score["auc"]:.3f}_loss_{loss:.3f}'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

        if rank == 0:
            torch.save(ddp_model.state_dict(), checkpoint_path)
            logger.info(f'[GPU {rank}] Checkpoint saved at {checkpoint_path}')

        dist.barrier()  # 저장이 완료될 때까지 대기

        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        ddp_model.load_state_dict(
            torch.load(checkpoint_path, map_location=map_location)
        )
        logger.info(f'[GPU {rank}] Checkpoint loaded from {checkpoint_path}')

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    wandb.login(key=args.wandb_key)
    wandb_run = wandb.init(project='outfit-transformer', config=args.__dict__)
    
    mp.spawn(
        train,
        args=(args.world_size, args, wandb_run),
        nprocs=args.world_size,
        join=True
    )
    
    train(args)
