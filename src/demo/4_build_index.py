import os
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sys
import glob

from ..models.load import load_model
from ..utils.utils import seed_everything
from ..demo.stores.vector import ItemVectorStore

import pathlib


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
LOADER_DIR = SRC_DIR / 'stores'
RESULT_DIR = SRC_DIR / 'results'

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--embedding_sz", type=int, default=128,
        help=""
    )
    return parser.parse_args()


def main(args):
    indexer = ItemVectorStore(
        index_name='polyvore',
        faiss_type='IndexFlatL2',
        base_dir=LOADER_DIR,
        d_embed=args.embedding_sz
    )

    pattern = 'embeddings_[0-9][0-9]_of_[0-9][0-9]*'
    matching_files = glob.glob(
        os.path.join(LOADER_DIR, pattern)
    )
    pbar = tqdm(
        matching_files, 
        desc="Shard", 
        postfix={"file": None}
    )
    for file_path in pbar:
        pbar.set_postfix(file=os.path.basename(file_path))
        with open(file_path, 'rb') as file:
            ids, embeddings = pickle.load(file)
            indexer.add(embeddings=embeddings, ids=ids)
    indexer.save()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)