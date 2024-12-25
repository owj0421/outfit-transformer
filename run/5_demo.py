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

from src.utils import slurm
from argparse import ArgumentParser

from src.model.load import (
    load_model
)

import sys
from fashion_recommenders.fashion_recommenders.data.loader import SQLiteItemLoader
from fashion_recommenders.fashion_recommenders.data.indexer import FAISSIndexer
from fashion_recommenders.fashion_recommenders.utils.demo import demo

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--task', type=str, required=True, choices=['cp', 'cir']
    )
    parser.add_argument(
        '--model_type', type=str, required=True, choices=['original', 'clip'], default='original'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None
    )
    parser.add_argument(
        "--db_dir", type=str, default="./src/db",
    )
    parser.add_argument(
        "--index_dir", type=str, default="./src/db",
    )
    return parser.parse_args()


def main(args):
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=IMAGE_DIR,
    )
    indexer = FAISSIndexer(
        index_dir=args.index_dir,
    )
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    )
    model.eval()
    demo(
        model=model,
        item_loader=loader,
        task='cir',
        indexer=indexer,
    )
    

if __name__ == "__main__":
    args = parse_args()
    main(args)