import os
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.datasets.load import (
    load_polyvore
)
from src.model.load import (
    load_model
)
from src.utils.utils import (
    seed_everything,
)
from src.utils.elements import (
    Item, 
    Outfit, 
    Query
)
from src.datasets.polyvore import PolyvoreItems
from tqdm import tqdm
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type', 
        type=str, choices=['original', 'clip'], default='clip'
    )
    parser.add_argument(
        '--polyvore_dir', 
        type=str, default="/home/owj0421/datasets/polyvore"
    )
    parser.add_argument(
        '--batch_sz', 
        type=int, default=128
    )
    parser.add_argument(
        '--seed', 
        type=int, default=42
    )
    parser.add_argument(
        '--save_dir',
        type=str, default="./index"
    )
    parser.add_argument(
        '--checkpoint',
        type=str, default=None
    )
    parser.add_argument(
        '--demo',
        action='store_true'
    )
    parser.add_argument(
        '--num_shards',
        type=int, default=1
    )
    parser.add_argument(
        '--shard_id',
        type=int, default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(
        args.seed
    )
    polyvore_items = PolyvoreItems(
        dataset_dir=args.polyvore_dir,
    )
    model = load_model(
        args
    )
    
    all_id, all_embedding = [], []
    batch_id, batch_item = [], []
    
    item_ids = list(polyvore_items.item_id_to_idx.keys())
    
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
                if len(all_id) == 10:
                    break
                
            item_id = item_ids[i]

            batch_id.append(item_id)
            batch_item.append(polyvore_items(item_id))
            
            if len(batch_id) == args.batch_sz or i == len(polyvore_items) - 1:
                batch_embedding = model.embed_item(batch_item)
                
                all_id.append(batch_id)
                all_embedding.append(batch_embedding)
                
                batch_id, batch_item = [], []
    
    all_id = sum(all_id, [])
    all_embedding = torch.cat(all_embedding, dim=0).detach().cpu().numpy()
    
    os.makedirs(
        args.save_dir, 
        exist_ok=True
    )
    save_file = os.path.join(
        args.save_dir, f"embeddings_{args.shard_id:02d}_of_{args.num_shards:02d}"
    )
    with open(save_file, mode="wb") as f:
        pickle.dump((all_id, all_embedding), f)
        
    print(
        f"Embeddings saved to {save_file}"
    )