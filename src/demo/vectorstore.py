# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

from tqdm import tqdm
    
    
import os
import faiss
import pathlib
from collections import defaultdict

from . import vectorstore_utils


class FAISSVectorStore:
    
    def __init__(
        self, 
        index_name: str = 'index',
        faiss_type: str = 'IndexFlatL2',
        base_dir: str = Path.cwd(),
        d_embed: int = 128,
        *faiss_args, **faiss_kwargs
    ):
        self.index_path = os.path.join(base_dir, f"{index_name}.faiss")
        
        if vectorstore_utils.faiss_exists(self.index_path):
            index = faiss.read_index(self.index_path)
        else:
            index = vectorstore_utils.create_faiss(faiss_type, d_embed, *faiss_args, **faiss_kwargs)
        
        self.index = index
        
        
    def add(
        self, 
        embeddings: List[List[float]], 
        ids: List[int],
        batch_size: int = 1000,
    ) -> None:
        return vectorstore_utils.add(self.index, embeddings, ids, batch_size)
            
            
    def search(
        self, 
        embeddings: List[List[float]],
        k: int,
        batch_size: int = 2048,
    ) -> List[Tuple[float, int]]:
        return vectorstore_utils.search(self.index, embeddings, k, batch_size)
    
    
    def save(self):
        vectorstore_utils.save(self.index, self.index_path)
        
        
    def multi_vector_search(
        self,
        embeddings: List[List[List[float]]],
        k: int,
        batch_size: int = 2048,
    ) -> List[List[int]]:
        """RRF Search
        """
        ids = []
        for es in embeddings: # es: (n_query_items, d_embed)
            scores = defaultdict(list)
            for result in self.search(es, 100, batch_size):
                for score, item_id in result:
                    scores[item_id].append(score)
            scores = {item_id: np.mean(score) for item_id, score in scores.items()}
            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True) # [(id, score), ...]
            ids.append(list(map(lambda x: x[0], scores))[:k])
            
        return ids