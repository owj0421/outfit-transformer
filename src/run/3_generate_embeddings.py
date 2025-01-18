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
from ..utils.utils import (
    seed_everything,
)
from tqdm import tqdm
import pickle
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
        type=str, choices=['original', 'clip'], default='clip'
    )
    parser.add_argument(
        '--batch_sz', 
        type=int, default=128
    )
    parser.add_argument(
        '--checkpoint',
        type=str, default=None
    )
    parser.add_argument(
        '--num_shards',
        type=int, default=1
    )
    parser.add_argument(
        '--shard_id',
        type=int, default=0
    )
    parser.add_argument(
        '--demo',
        action='store_true'
    )
    return parser.parse_args()


def main(args):
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    )
    
    all_ids, all_embeddings = [], []
    batch_id, batch_item = [], []
    
    item_ids = loader.conn.execute("SELECT item_id FROM items").fetchall()
    
    shard_size = len(item_ids) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    
    if args.shard_id == args.num_shards - 1:
        end_idx = len(item_ids)

    item_ids = item_ids[start_idx:end_idx]
    print(f"Embedding generation for {len(item_ids)} passages from idx {start_idx} to {end_idx}.")
    
    with torch.no_grad():
        for i in tqdm(range(len(item_ids))):
            if args.demo:
                if len(all_ids) == 10:
                    break
                
            item_id = item_ids[i][0]

            batch_id.append(item_id)
            batch_item.append(loader.get_item(item_id))
            
            if len(batch_id) == args.batch_sz or i == len(item_ids) - 1:
                batch_embedding = model.embed_item(batch_item)
                
                all_ids.append(batch_id)
                all_embeddings.append(batch_embedding)
                
                batch_id, batch_item = [], []
    
    all_ids = sum(all_ids, [])
    all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
    
    os.makedirs(
        LOADER_DIR, exist_ok=True
    )
    save_file = os.path.join(
        LOADER_DIR, f"embeddings_{args.shard_id:02d}_of_{args.num_shards:02d}"
    )
    with open(save_file, mode="wb") as f:
        pickle.dump((all_ids, all_embeddings), f)
        
    print(
        f"Embeddings saved to {save_file}"
    )
    
    


if __name__ == "__main__":
    args = parse_args()
    
    main(args)