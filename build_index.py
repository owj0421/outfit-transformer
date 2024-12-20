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

from src.index.indexer import Indexer
from src.utils import slurm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(args):
    os.makedirs(
        args.save_dir, 
        exist_ok=True
    )
     
    print(
        f"Init Indexer at {args.save_dir}.", flush=True
    )
    indexer = Indexer(
        embedding_sz=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits,
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
        print(
            f"Indexing passages from {file_path}..."
        )
        pbar.set_postfix(
            file=os.path.basename(file_path)
        )
        with open(file_path, 'rb') as file:
            all_id, all_embedding = pickle.load(file)
            indexer.index_with_ids(
                all_id, all_embedding
            )
    
    print(
        f"Saving index to {args.save_dir}."
    )
    indexer.save_local(
        database_path=args.save_dir,
        override=True
    )
    
    print(
        f"Complete!"
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection_size", type=int, default=128,
        help=""
    )
    parser.add_argument(
        "--n_subquantizers", type=int, default=0,
        help="Number of subquantizer used for embedding quantization, if 0 flat index is used"
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, 
        help="Number of bits per subquantizer"
    )
    parser.add_argument(
        "--embeddings_dir", type=str,
        help="dir path to embeddings"
    )
    parser.add_argument(
        "--save_dir", type=str,
        help="dir path to save index"
    )
    args = parser.parse_args()
    slurm.init_distributed_mode(args)
    main(args)