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
        '--task', 
        type=str, required=True, choices=['cp', 'cir']
    )
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
        '--batch_sz', 
        type=int, default=32
    )
    parser.add_argument(
        '--n_workers', 
        type=int, default=4
    )
    parser.add_argument(
        '--n_epochs', 
        type=int, default=4
    )
    parser.add_argument(
        '--lr', 
        type=float, default=4e-5
    )
    parser.add_argument(
        '--accumulation_steps', 
        type=int, default=1
    )
    parser.add_argument(
        '--wandb_key', 
        type=str, default=None
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


def cp_train(args):
    metric = CompatibilityMetricCalculator()
    
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    train = polyvore.PolyvoreCompatibilityDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='train'
    )
    valid = polyvore.PolyvoreCompatibilityDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='valid'
    )
    train_dataloader = DataLoader(
        dataset=train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=train.collate_fn
    )
    valid_dataloader = DataLoader(
        dataset=valid,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=valid.collate_fn
    )
    

    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,  # 최대 학습률
        epochs=args.n_epochs,  # 총 에폭
        steps_per_epoch=len(train_dataloader),  # 에폭당 스텝 수
        pct_start=0.3,  # 최대 학습률 도달 비율 (30%)
        anneal_strategy='cos',  # 코사인 감소
        div_factor=25,  # 초기 학습률 = max_lr / div_factor
        final_div_factor=1e4  # 최종 학습률 = max_lr / final_div_factor
    )
    loss_fn = focal_loss
    
    n_train_steps = 0
    n_eval_steps = 0
    
    for epoch in range(args.n_epochs):
        model.train()
        
        pbar = tqdm(
            train_dataloader,
            desc=f'[Train] Epoch {epoch+1}/{args.n_epochs}'
        )
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
            n_train_steps += 1
            
            labels = torch.FloatTensor(data['label']).cuda()
            queries = data['query']
            
            predictions = model.predict(queries=queries).squeeze()
            
            loss = loss_fn(y_prob=predictions, y_true=labels) 
            loss = loss / args.accumulation_steps
            loss.backward()
            
            if (n_train_steps + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() 
            if scheduler:
                scheduler.step()
            
            score = metric.add(
                predictions=predictions.detach().cpu().numpy(),
                labels=labels.detach().cpu().numpy()
            )
            score = {
                f'train_{key}': value
                for key, value in score.items()
            }
            if args.wandb_key:
                log = {
                    'train_loss': loss * args.accumulation_steps,
                    'n_train_steps': n_train_steps,
                    'lr': float(scheduler.get_last_lr()[0]) if scheduler else args.lr,
                    **score
                }
                wandb.log(log)
            pbar.set_postfix(
                loss=loss.item() * args.accumulation_steps,
                **score
            )
        score = metric.calculate()
        print(f'[Train] Epoch {epoch+1}/{args.n_epochs} --> {score}')
        
        
        model.eval()
        pbar = tqdm(
            valid_dataloader,
            desc=f'[Valid] Epoch {epoch+1}/{args.n_epochs}'
        )
        with torch.no_grad():
            for i, data in enumerate(pbar):
                if args.demo and i > 10:
                    break
                n_train_steps += 1
                
                labels = torch.FloatTensor(data['label']).cuda()
                queries = data['query']
                
                predictions = model.predict(queries=queries).squeeze()
            
                loss = loss_fn(y_prob=predictions, y_true=labels) 
                loss = loss / args.accumulation_steps
            
                score = metric.add(
                    predictions=predictions.detach().cpu().numpy(),
                    labels=labels.detach().cpu().numpy()
                )
                score = {
                    f'valid_{key}': value
                    for key, value in score.items()
                }
                if args.wandb_key:
                    log = {
                        'valid_loss': loss * args.accumulation_steps,
                        'n_valid_steps': n_eval_steps,
                        **score
                    }
                    wandb.log(log)
                pbar.set_postfix(
                    loss=loss.item() * args.accumulation_steps,
                    **score
                )
                
        score = metric.calculate()
        print(f'[Valid] Epoch {epoch+1}/{args.n_epochs} --> {score}')


        save_dir = os.path.join(
            CHECKPOINT_DIR, 
            f"{args.task}-{args.model_type}-{wandb.run.name}", 
            f'epoch_{epoch+1}_acc_{score["acc"]:.3f}_auc_{score["auc"]:.3f}_loss_{loss:.3f}'
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f)
        
        torch.save(
            obj=model.state_dict(),
            f=os.path.join(save_dir, f'model.pt')
        )
        if optimizer:
            torch.save(
                obj=optimizer.state_dict(),
                f=os.path.join(save_dir, f'optimizer.pt')
            )
        if scheduler:
            torch.save(
                obj=scheduler.state_dict(),
                f=os.path.join(save_dir, f'scheduler.pt')
            )
        print(f'[     ] Epoch {epoch+1}/{args.n_epochs} --> Saved at {save_dir}')


def cir_train(args):       
    metric = ComplementaryMetricCalculator()
    
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    train = polyvore.PolyvoreTripletDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='train'
    )
    valid = polyvore.PolyvoreTripletDataset(
        loader, 
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='valid'
    )
    train_dataloader = DataLoader(
        dataset=train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=train.collate_fn
    )
    valid_dataloader = DataLoader(
        dataset=valid,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=valid.collate_fn
    )
    
    # Load Model
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,  # 최대 학습률
        epochs=args.n_epochs,  # 총 에폭
        steps_per_epoch=len(train_dataloader),  # 에폭당 스텝 수
        pct_start=0.3,  # 최대 학습률 도달 비율 (30%)
        anneal_strategy='cos',  # 코사인 감소
        div_factor=25,  # 초기 학습률 = max_lr / div_factor
        final_div_factor=1e4  # 최종 학습률 = max_lr / final_div_factor
    )
    loss_fn = torch.nn.TripletMarginLoss(margin=2.0, p=2)
    
    n_train_steps = 0
    n_eval_steps = 0
    
    for epoch in range(args.n_epochs):
        model.train()
        pbar = tqdm(
            train_dataloader,
            desc=f'[Train] Epoch {epoch+1}/{args.n_epochs}'
        )
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
            n_train_steps += 1
            
            labels = np.zeros(args.batch_sz)

            queries = data['query']
            query_embedddings = model.embed_query(
                queries=queries
            )
            
            answers = data['answer']
            answer_embedddings = model.embed_item(
                items=answers
            ) # (batch_sz, embedding_dim)
            
            n_candidates = args.batch_sz # 1 positive + n-1 negative
            
            negative_embeddings = []
            for i in range(args.batch_sz):
                in_batch_negatives = [
                    answer_embedddings[j, :] for j in range(args.batch_sz) if i != j
                ]
                negative_embeddings.append(torch.stack(in_batch_negatives))

            candidate_embedddings = []
            for i in range(args.batch_sz):
                candidate_embedddings.append(
                    torch.cat(
                        [answer_embedddings[i].unsqueeze(0), negative_embeddings[i]],
                        dim=0
                    )
                )
            
            loss = 0
            for b_i, (qs, cs) in enumerate(zip(query_embedddings, candidate_embedddings)):
                for q_i in range(qs.shape[0]):
                    loss += loss_fn(
                        qs[q_i, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                        cs[0, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                        cs[1:, :] # (n_candidates-1, embedding_dim)
                    )
            loss = loss / (args.accumulation_steps * (b_i+1) * (q_i+1))
            
            score = metric.add(
                query_embeddings=[q.detach().cpu().numpy() for q in query_embedddings],
                candidate_embeddings=[c.detach().cpu().numpy() for c in candidate_embedddings],
                labels=labels
            )
            score = {
                f'valid_{key}': value
                for key, value in score.items()
            }
            if args.wandb_key:
                log = {
                    'train_loss': loss * args.accumulation_steps,
                    'n_train_steps': n_train_steps,
                    'lr': float(scheduler.get_last_lr()[0]) if scheduler else args.lr,
                    **score
                }
                wandb.log(log)
            pbar.set_postfix(
                loss=loss.item() * args.accumulation_steps,
                **score
            )
            
        score = metric.calculate()
        print(f'[Train] Epoch {epoch+1}/{args.n_epochs} --> {score}')

        model.eval()
        pbar = tqdm(
            valid_dataloader,
            desc=f'[Valid] Epoch {epoch+1}/{args.n_epochs}'
        )
        with torch.no_grad():
            for i, data in enumerate(pbar):
                if args.demo and i > 10:
                    break
                n_eval_steps += 1
                
                labels = np.zeros(args.batch_sz)
                
                queries = data['query']
                query_embedddings = model.embed_query(
                    queries=queries
                )
                
                answers = data['answer']
                answer_embedddings = model.embed_item(
                    items=answers
                )
                
                negative_embeddings = []
                for i in range(args.batch_sz):
                    in_batch_negatives = [
                        answer_embedddings[j, :] for j in range(args.batch_sz) if i != j
                    ]
                    negative_embeddings.append(torch.stack(in_batch_negatives))

                candidate_embedddings = []
                for i in range(args.batch_sz):
                    candidate_embedddings.append(
                        torch.cat(
                            [answer_embedddings[i].unsqueeze(0), negative_embeddings[i]],
                            dim=0
                        )
                    )
                
                loss = 0
                for b_i, (qs, cs) in enumerate(zip(query_embedddings, candidate_embedddings)):
                    for q_i in range(qs.shape[0]):
                        loss += loss_fn(
                            qs[q_i, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                            cs[0, :].unsqueeze(0).expand(n_candidates - 1, -1), # (n_candidates, embedding_dim)
                            cs[1:, :] # (n_candidates-1, embedding_dim)
                        )
                loss = loss / (args.accumulation_steps * (b_i+1) * (q_i+1))
                
                score = metric.add(
                    query_embeddings=[q.detach().cpu().numpy() for q in query_embedddings],
                    candidate_embeddings=[c.detach().cpu().numpy() for c in candidate_embedddings],
                    labels=labels
                )
                score = {
                    f'valid_{key}': value
                    for key, value in score.items()
                }
                if args.wandb_key:
                    log = {
                        'valid_loss': loss * args.accumulation_steps,
                        'n_eval_steps': n_eval_steps,
                        **score
                    }
                    wandb.log(log)
                pbar.set_postfix(
                    loss=loss.item() * args.accumulation_steps,
                    **score
                )
                
        score = metric.calculate()
        print(
            f"[Valid] Epoch {epoch+1}/{args.n_epochs} --> {score}"
        )
        
        save_dir = os.path.join(
            CHECKPOINT_DIR, 
            f"{args.task}-{args.model_type}-{wandb.run.name}", 
            f'epoch_{epoch+1}_acc_{score["acc"]:.3f}_loss_{loss:.3f}'
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f)
        
        torch.save(
            obj=model.state_dict(),
            f=os.path.join(save_dir, f'model.pt')
        )
        if optimizer:
            torch.save(
                obj=optimizer.state_dict(),
                f=os.path.join(save_dir, f'optimizer.pt')
            )
        if scheduler:
            torch.save(
                obj=scheduler.state_dict(),
                f=os.path.join(save_dir, f'scheduler.pt')
            )
        print(f'[     ] Epoch {epoch+1}/{args.n_epochs} --> Saved at {save_dir}')


if __name__ == '__main__':
    args = parse_args()
    wandb.login(
        key=args.wandb_key
    )
    wandb.init(
        project='outfit-transformer',
        config=args.__dict__,
    )
    seed_everything(
        args.seed
    )
    if args.task == 'cp':
        cp_train(args)
    elif args.task == 'cir':
        cir_train(args)