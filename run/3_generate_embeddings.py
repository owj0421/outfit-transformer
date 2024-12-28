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
from tqdm import tqdm
import pickle
import sys
from fashion_recommenders.data.loader import SQLiteItemLoader

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
        "--db_dir", type=str, default="./src/db",
        help="dir path to save index"
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str, default="./src/db"
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


if __name__ == "__main__":
    args = parse_args()
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    )
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
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
            batch_item.append(loader(item_id))
            
            if len(batch_id) == args.batch_sz or i == len(item_ids) - 1:
                batch_embedding = model.embed_item(batch_item)
                
                all_ids.append(batch_id)
                all_embeddings.append(batch_embedding)
                
                batch_id, batch_item = [], []
    
    all_ids = sum(all_ids, [])
    all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
    
    os.makedirs(
        args.embeddings_dir, 
        exist_ok=True
    )
    save_file = os.path.join(
        args.embeddings_dir, f"embeddings_{args.shard_id:02d}_of_{args.num_shards:02d}"
    )
    with open(save_file, mode="wb") as f:
        pickle.dump((all_ids, all_embeddings), f)
        
    print(
        f"Embeddings saved to {save_file}"
    )