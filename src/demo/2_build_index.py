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

from . import vectorstore
from ..data import collate_fn
from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, setup
from ..utils.logger import get_logger
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    
    return parser.parse_args()


def load_rec_embedding_dict(dataset_dir):
    e_dir = POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=dataset_dir)
    filenames = [filename for filename in os.listdir(e_dir) if filename.endswith(".pkl")]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    
    all_ids, all_embeddings = [], []
    for filename in filenames:
        filepath = os.path.join(e_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_ids += data['ids']
            all_embeddings.append(data['embeddings'])
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Loaded {len(all_embeddings)} embeddings")
    
    all_embeddings_dict = {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}
    print(f"Created embeddings dictionary")
    
    return all_embeddings_dict


def main(args):
    indexer = vectorstore.FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir),
    )
    rec_embedding_dict = load_rec_embedding_dict(args.polyvore_dir)
    
    embeddings = list(rec_embedding_dict.values())
    ids = list(rec_embedding_dict.keys())
    
    indexer.add(embeddings=embeddings, ids=ids)
    
    indexer.save()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)