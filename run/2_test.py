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
from src.utils.utils import (
    seed_everything,
)
import sys
from fashion_recommenders.utils.elements import Item, Outfit, Query
from fashion_recommenders.utils.metrics import score_cp, score_fitb
from fashion_recommenders.data.loader import SQLiteItemLoader
from fashion_recommenders.data.datasets import polyvore

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = ArgumentParser()
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
        '--result_dir',
        type=str, default=None, default='./src/results'
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
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=os.path.join(args.polyvore_dir, 'images'),
    )
    test = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='train'
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=test.collate_fn
    )
    
    # Load Model
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    model.eval()
    
    pbar = tqdm(
        test_dataloader,
        desc=f'CP Test'
    )
    n_test_steps = 0
    all_pred, all_label = [], []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
        
            n_test_steps += 1
            
            label = torch.FloatTensor(
                data['label']
            ).cuda()
            outfits = [
                Outfit(
                    items=[loader(int(item_id)) for item_id in batch]
                )
                for batch in data['question']
            ]
            pred = model.predict(
                outfits=outfits
            ).squeeze()
            score = score_cp(
                pred=pred,
                label=label,
                compute_auc=False
            )
            all_pred.append(
                pred.detach().cpu()
            )
            all_label.append(
                label.cpu()
            )
            pbar.set_postfix(
                acc=score["acc"] * 100
            )
            
    all_pred = torch.cat(
        all_pred
    )
    all_label = torch.cat(
        all_label
    )
    score = score_cp(
        pred=all_pred,
        label=all_label,
        compute_auc=True
    )
    print(
        f'--> Test Acc: {score["acc"]:.3f} | Test AUC: {score["auc"]:.3f}'
    )
    
    if args.result_dir is not None:
        result_dir = os.path.join(
            args.result_dir, args.checkpoint.split('/')[-3], args.checkpoint.split('/')[-2],
        )
        os.makedirs(
            result_dir, exist_ok=True
        )
        with open(os.path.join(result_dir, 'cp_predictions.json'), 'w') as f:
            json.dump(
                all_pred.tolist(), f
            )
        with open(os.path.join(result_dir, 'cp_labels.json'), 'w') as f:
            json.dump(
                all_label.tolist(), f
            )
        with open(os.path.join(result_dir, 'cp_results.json'), 'w') as f:
            json.dump(
                {"acc": score["acc"], "auc": score["auc"]}, f
            )
        print(
            f"--> Results saved to {result_dir}"
        )


def fitb_valid(args):
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=os.path.join(args.polyvore_dir, 'images'),
    )
    test = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir,
        polyvore_type=args.polyvore_type,
        split='train'
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=test.collate_fn
    )
    
    # Load Model
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    ) 
    model.eval()
    pbar = tqdm(
        test_dataloader,
        desc=f'FITB Test'
    )
    n_test_steps = 0
    all_pred, all_label = [], []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 10:
                break
        
            n_test_steps += 1
            
            label = torch.Tensor(data['label']).cuda() # List of [batch_sz]
        
            batch_sz = len(data['question'])
            n_candidates = len(data['candidates'][0])
            
            query = [
                Query(query=loader.get_category(int(cs[int(l)])),
                        items=[loader(item_id) for item_id in q])
                for q, cs, l in zip(data['question'], data['candidates'], data['label'])
            ]
            candidate_outfits = [
                loader(int(batch[i]))
                for batch in data['candidates'] for i in range(n_candidates)
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
            pbar.set_postfix(
                acc=score["acc"] * 100
            )
            
    all_pred = torch.cat(all_pred)
    all_label = torch.cat(all_label)
    acc = (all_pred == all_label).sum().item() / len(all_label)
    print(
        f"--> Test Acc: {acc:.3f}"
    )
    
    if args.result_dir is not None:
        result_dir = os.path.join(
            args.result_dir, args.checkpoint.split('/')[-3], args.checkpoint.split('/')[-2],
        )
        os.makedirs(
            result_dir, exist_ok=True
        )
        with open(os.path.join(result_dir, 'fitb_predictions.json'), 'w') as f:
            json.dump(
                all_pred.tolist(), f
            )
        with open(os.path.join(result_dir, 'fitb_labels.json'), 'w') as f:
            json.dump(
                all_label.tolist(), f
            )
        with open(os.path.join(result_dir, 'fitb_results.json'), 'w') as f:
            json.dump(
                {"acc": acc}, f
            )
        print(
            f"--> Results saved to {result_dir}"
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