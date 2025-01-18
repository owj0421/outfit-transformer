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
from ..pipeline import OutfitTransformerPipeline
from fashion_recommenders.stores.metadata import ItemMetadataStore
from fashion_recommenders.stores.vector import ItemVectorStore
from fashion_recommenders import demo

import pathlib


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
LOADER_DIR = SRC_DIR / 'stores'
RESULT_DIR = SRC_DIR / 'results'

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
    return parser.parse_args()


def main(args):
    model = load_model(
        model_type=args.model_type,
        checkpoint=args.checkpoint
    )
    model.eval()
    
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    indexer = ItemVectorStore(
        index_name='polyvore',
        faiss_type='IndexFlatL2',
        base_dir=LOADER_DIR,
        d_embed=128
    )

    pipeline = OutfitTransformerPipeline(
        model=model,
        loader=loader,
        indexer=indexer
    )
        
    demo.run(pipeline=pipeline, task=args.task)

if __name__ == "__main__":
    args = parse_args()
    main(args)