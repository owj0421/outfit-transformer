import os
import argparse
import json
import pickle
import time
import glob
from tqdm import tqdm
import os

import argparse
import pickle

from ..utils import slurm
from argparse import ArgumentParser

import sys
from fashion_recommenders.data.indexer import FAISSIndexer
from fashion_recommenders.data.loader import SQLiteItemLoader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--embedding_sz", type=int, default=128,
        help=""
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="./src/db",
        help="dir path to embeddings"
    )
    parser.add_argument(
        "--index_dir", type=str, default="./src/db",
        help="dir path to save index"
    )
    return parser.parse_args()


def main(args):
    indexer = FAISSIndexer(
        index_dir=args.index_dir,
        embedding_sz=args.embedding_sz
    )

    pattern = 'embeddings_[0-9][0-9]_of_[0-9][0-9]*'
    matching_files = glob.glob(
        os.path.join(
            args.embeddings_dir, pattern
        )
    )
    pbar = tqdm(
        matching_files, 
        desc="Shard", 
        postfix={"file": None}
    )
    for file_path in pbar:
        pbar.set_postfix(
            file=os.path.basename(file_path)
        )
        with open(file_path, 'rb') as file:
            ids, embeddings = pickle.load(file)
            indexer.build(
                ids=ids, 
                embeddings=embeddings
            )
    indexer.save()
    

if __name__ == "__main__":
    args = parse_args()
    slurm.init_distributed_mode(args)
    main(args)