import os
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
import json
import torch

class Indexer(object):

    def __init__(
        self, 
        embedding_sz: int = 128,
        n_subquantizers: int = 0,
        n_bits: int = 8,
        index: Optional[faiss.Index] = None,
    ):
        self.cfg = {
            "embedding_sz": embedding_sz,
            "n_subquantizers": n_subquantizers,
            "n_bits": n_bits
        }
        
        if not index:
            if n_subquantizers > 0:
                index = faiss.IndexPQ(
                    embedding_sz, 
                    n_subquantizers, 
                    n_bits, 
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                index = faiss.IndexFlatIP(
                    embedding_sz
                )
                index = faiss.IndexIDMap2(index)
                
        self.index = index
    
    @classmethod
    def load_local(
        cls,
        database_path: str = "./src/index/index",
    ):
        index_path = os.path.join(
            database_path, 'index.faiss'
        )
        args_path = os.path.join(
            database_path, 'index_args.json'
        )
        with open(args_path, 'r') as f:
            args = json.load(f)
        obj = cls(
            index=faiss.read_index(index_path),
            **args
        )
        print(
            f"[{'Indexer':^16}] Loaded from {database_path}.", 
            end=" "
        )
        
        return obj
            
            
    def save_local(
        self,
        database_path: str = "./src/index/index",
        override: bool = False
    ) -> None:
        index_path = os.path.join(
            database_path, 'index.faiss'
        )
        args_path = os.path.join(
            database_path, 'index_args.json'
        )
        if os.path.exists(index_path) and override is False:
            raise FileExistsError(
                f"[{'Indexer':^16}] Index file already exists at {database_path}. Set 'override=True' to overwrite."
            )
        print(
            f"[{'Indexer':^16}] Serializing Index to {database_path}...", 
            end=" "
        )
        faiss.write_index(
            self.index, index_path
        )
        with open(args_path, 'w') as f:
            json.dump(
                self.cfg, f
            )
        print(
            f"Serializing Complete!"
        )

    def index_with_ids(
        self, 
        ids: List[int],
        embeddings: np.array
    ):
        if not self.index.is_trained:
            self.index.train(embeddings)
            
        embeddings = embeddings.astype('float32')
        pbar = tqdm(
            zip(ids, embeddings),
            desc=f"[FAISS] Indexing", total=len(ids)
        )
        for id, embedding in pbar:
            self.index.add_with_ids(
                embedding[np.newaxis, :], np.array([int(id)])
            )
        print(
            f"[{'Indexer':^16}] Build Complete! Total {len(ids)}"
        )

    def search(
        self, 
        query_embeddings: np.array, 
        top_k: int = 8, 
        index_batch_size: int = 2048,
    ):
        
        assert query_embeddings.dtype == np.float32, (
            f"[{'Indexer':^16}] Vectors must be of type float32"
        )
        
        total_batches = (len(query_embeddings) + index_batch_size - 1) // index_batch_size  # 전체 배치 수 계산
        pbar = tqdm(
            range(0, len(query_embeddings), index_batch_size), 
            desc="[FAISS] Searching", total=total_batches
        )
        outputs = []
        for i in pbar:
            embeddings = query_embeddings[i:i + index_batch_size]
            # check dim and keep dim 2
            if len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
                
            batch_scores, batch_faiss_ids = self.index.search(
                embeddings, k=top_k
            )
            for scores, faiss_ids in zip(batch_scores, batch_faiss_ids):
                outputs.append([
                    {
                        'id': str(faiss_id),
                        'score': score
                    } for faiss_id, score in zip(faiss_ids, scores)
                ])
            
        return outputs