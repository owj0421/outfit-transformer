import os
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from src.model.load import (
    load_model
)
from src.utils.loss import (
    focal_loss,
)
from src.utils.utils import (
    seed_everything,
)

import sys
from fashion_recommenders.fashion_recommenders.utils.elements import Item, Outfit, Query
from fashion_recommenders.fashion_recommenders.utils.metrics import score_cp, score_fitb
from fashion_recommenders.fashion_recommenders.data.loader import SQLiteItemLoader
from fashion_recommenders.fashion_recommenders.data.datasets import polyvore


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
        '--db_dir', 
        type=str, required=True, default='./src/db'
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
        '--save_dir',
        type=str, default=None, default='./src/checkpoints'
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
    save_dir = os.path.join(
        args.save_dir, f"{args.task}-{args.model_type}-{wandb.run.name}"
    )
    
    # Load Data
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=os.path.join(args.polyvore_dir, 'images'),
    )
    train, valid = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='train'
    ), polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='valid'
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
    loss_fn = focal_loss
    
    n_train_steps = 0
    n_eval_steps = 0
    
    for epoch in range(args.n_epochs):
        model.train()
        pbar = tqdm(
            train_dataloader,
            desc=f'Train | Epoch {epoch+1}/{args.n_epochs}'
        )
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
            n_train_steps += 1
            
            label = torch.FloatTensor(data['label']).cuda()
            outfits = [
                Outfit(items=[loader(int(item_id)) for item_id in batch])
                for batch in data['question']
            ]
            pred = model.predict(
                outfits=outfits
            ).squeeze()
            loss = loss_fn(
                y_prob=pred, 
                y_true=label
            ) / args.accumulation_steps
            score = score_cp(
                pred=pred,
                label=label,
                compute_auc=False
            )
            loss.backward()
            
            if (n_train_steps + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() 
                
            if scheduler:
                scheduler.step()
                
            if args.wandb_key:
                log = {
                    'train_loss': loss * args.accumulation_steps,
                    'train_acc': score["acc"],
                    'n_train_steps': n_train_steps,
                    'lr': float(scheduler.get_last_lr()[0]) if scheduler else args.lr
                }
                wandb.log(log)
            pbar.set_postfix(
                loss=loss.item() * args.accumulation_steps,
                acc=score["acc"] * 100
            )

        model.eval()
        pbar = tqdm(
            valid_dataloader,
            desc=f'Valid | Epoch {epoch+1}/{args.n_epochs}'
        )
        all_pred, all_label = [], []
        with torch.no_grad():
            for i, data in enumerate(pbar):
                if args.demo and i > 10:
                    break
                n_eval_steps += 1
                
                label = torch.FloatTensor(data['label']).cuda()
                outfits = [
                    Outfit(items=[loader(item_id) for item_id in q])
                    for q in data['question']
                ]
                pred = model.predict(
                    outfits=outfits
                ).squeeze()
                loss = loss_fn(
                    y_prob=pred,
                    y_true=label
                ) / args.accumulation_steps
                score = score_cp(
                    pred=pred,
                    label=label,
                    compute_auc=False
                )
                all_pred.append(pred.detach().cpu())
                all_label.append(label.detach().cpu())
                if args.wandb_key:
                    log = {
                        'valid_loss': loss * args.accumulation_steps,
                        'valid_acc': score["acc"],
                        'n_eval_steps': n_eval_steps,
                    }
                    wandb.log(log)
                pbar.set_postfix(
                    loss=loss.item() * args.accumulation_steps,
                    acc=score["acc"] * 100
                )
                
        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        score = score_cp(
            pred=all_pred,
            label=all_label,
            compute_auc=True
        )
        loss = loss_fn(
            y_prob=all_pred, 
            y_true=all_label
        )
        print(
            f'--> Epoch {epoch+1}/{args.n_epochs} | Valid Loss: {loss:.3f} | Valid Acc: {score["acc"]:.3f} | Valid AUC: {score["auc"]:.3f}'
        )
        if args.save_dir:
            epoch_save_dir = os.path.join(
                save_dir, f'epoch_{epoch+1}_acc_{score["acc"]:.3f}_auc_{score["auc"]:.3f}_loss_{loss:.3f}'
            )
            if not os.path.exists(epoch_save_dir):
                os.makedirs(epoch_save_dir, exist_ok=True)
                
            with open(os.path.join(epoch_save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f)
            
            torch.save(
                obj=model.state_dict(),
                f=os.path.join(epoch_save_dir, f'model.pt')
            )
            if optimizer:
                torch.save(
                    obj=optimizer.state_dict(),
                    f=os.path.join(epoch_save_dir, f'optimizer.pt')
                )
            if scheduler:
                torch.save(
                    obj=scheduler.state_dict(),
                    f=os.path.join(epoch_save_dir, f'scheduler.pt')
                )
            print(
                f'--> Saved at {epoch_save_dir}'
            )


def cir_train(args):
    n_all_samples = 10
    n_hard_samples = 10
            
    save_dir = os.path.join(
        args.save_dir, f"{args.task}-{args.model_type}-{wandb.run.name}"
    )
    
    # Load Data
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=os.path.join(args.polyvore_dir, 'images'),
    )
    train, valid = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='train'
    ), polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='valid'
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
            desc=f'Train | Epoch {epoch+1}/{args.n_epochs}'
        )
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
            
            n_train_steps += 1

            batch_sz = len(data['anchor'])
            
            query = [
                Query(
                    query=loader.get_category(int(p[0])),
                    items=[loader(int(item_id)) for item_id in a]
                )
                for a, p in zip(data['anchor'], data['positive'])
            ]
            pos = [
                loader(int(p[0]))
                for p in data['positive']
            ]
            all_negs = [
                negative_item
                for _ in data['positive'] 
                for negative_item in loader.sample_items(n_all_samples)
            ]
            hard_negs = [
                negative_item
                for p in data['positive'] 
                for negative_item in loader.sample_items_by_id(int(p[0]), n_hard_samples)
            ]

            query_embed = model.embed_query(
                query=query
            ) # [batch_sz, dim]
            pos_embed = model.embed_item(
                item=pos
            ) # [batch_sz, dim]
            all_neg_embeds = torch.cat(
                [
                    model.embed_item(
                        item=all_negs[i*batch_sz:(i+1)*batch_sz]
                    )
                    for i in range(n_all_samples)
                ],
                dim=0
            ).view(batch_sz, n_all_samples, -1)
            hard_neg_embeds = torch.cat(
                [
                    model.embed_item(
                        item=hard_negs[i*batch_sz:(i+1)*batch_sz]
                    )
                    for i in range(n_hard_samples)
                ],
                dim=0
            ).view(batch_sz, n_hard_samples, -1)
            candidate_embeds = torch.cat(
                [pos_embed.unsqueeze(1), all_neg_embeds, hard_neg_embeds],
                dim=1
            ) # [batch_sz, 1 + n_all_samples + n_hard_samples, dim]
            
            batch_sz, n_candidates, d_embed = candidate_embeds.size()
            
            loss = 0
            for i in range(1, n_candidates):
                loss += loss_fn(
                    query_embed,
                    candidate_embeds[:, 0, :],
                    candidate_embeds[:, i, :]
                ) / (n_candidates - 1)
            
            loss = loss / args.accumulation_steps
            loss.backward()
            
            with torch.no_grad():
                # anchor - positive거리가 가장 짧은지 측정
                _, score = score_fitb(
                    query_embed=query_embed,
                    candidate_embeds=candidate_embeds,
                    label=torch.zeros(batch_sz).cuda()
                )
            
            if (n_train_steps + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if scheduler:
                scheduler.step()
                
            if args.wandb_key:
                log = {
                    'train_loss': loss * args.accumulation_steps,
                    'n_train_steps': n_train_steps,
                    "acc": score["acc"],
                    'lr': float(scheduler.get_last_lr()[0]) if scheduler else args.lr
                }
                wandb.log(log)
            pbar.set_postfix(
                loss=loss.item() * args.accumulation_steps,
                acc=score["acc"] * 100
            )

        model.eval()
        pbar = tqdm(
            valid_dataloader,
            desc=f'Valid | Epoch {epoch+1}/{args.n_epochs}'
        )
        all_pred, all_label = [], []
        with torch.no_grad():
            for i, data in enumerate(pbar):
                if args.demo and i > 10:
                    break
            
                n_eval_steps += 1
                
                label = torch.Tensor(data['label']).cuda() # List of [batch_sz]
            
                batch_sz = len(data['question'])
                n_candidates = len(data['candidates'][0])
                
                query = [
                    Query(query=loader.get_category(int(cs[int(l)])),
                          items=[loader(item_id) for item_id in q])
                    for q, cs, l in zip(data['question'], data['candidates'], data['label'])
                ]
                candidate_outfits = [
                    loader(c)
                    for cs in data['candidates'] for c in cs
                ]
                query_embed = model.embed_query(
                    query=query
                ) # [batch_sz, dim]
                candidate_embeds = torch.cat(
                    [
                        model.embed_item(
                            item=candidate_outfits[i*batch_sz:(i+1)*batch_sz]
                        )
                        for i in range(n_candidates)
                    ],
                    dim=0
                ).view(batch_sz, n_candidates, -1)
                
                loss = 0
                for i in range(1, n_candidates):
                    loss += loss_fn(
                        query_embed,
                        candidate_embeds[:, 0, :],
                        candidate_embeds[:, i, :]
                    ) / (n_candidates - 1)
                
                loss = loss / args.accumulation_steps
                
                pred, score = score_fitb(
                    query_embed=query_embed,
                    candidate_embeds=candidate_embeds,
                    label=label.cuda()
                )
                all_pred.append(
                    pred.detach().cpu()
                )
                all_label.append(
                    label.detach().cpu()
                )
                if args.wandb_key:
                    log = {
                        'valid_loss': loss * args.accumulation_steps,
                        'valid_acc': score["acc"],
                        'n_eval_steps': n_eval_steps,
                    }
                    wandb.log(log)
                pbar.set_postfix(
                    loss=loss.item() * args.accumulation_steps,
                    acc=score["acc"] * 100
                )
                
        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        acc = (all_pred == all_label).sum().item() / len(all_label)
        print(
            f"--> Epoch {epoch+1}/{args.n_epochs} | Valid FITB Acc: {acc:.3f}"
        )
        if args.save_dir:
            epoch_save_dir = os.path.join(
                save_dir, f'epoch_{epoch+1}_acc_{acc:.3f}_loss_{loss:.3f}'
            )
            if not os.path.exists(epoch_save_dir):
                os.makedirs(epoch_save_dir, exist_ok=True)
                
            with open(os.path.join(epoch_save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f)
            
            torch.save(
                obj=model.state_dict(),
                f=os.path.join(epoch_save_dir, f'model.pt')
            )
            if optimizer:
                torch.save(
                    obj=optimizer.state_dict(),
                    f=os.path.join(epoch_save_dir, f'optimizer.pt')
                )
            if scheduler:
                torch.save(
                    obj=scheduler.state_dict(),
                    f=os.path.join(epoch_save_dir, f'scheduler.pt')
                )
            print(
                f'--> Saved at {epoch_save_dir}'
            )


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