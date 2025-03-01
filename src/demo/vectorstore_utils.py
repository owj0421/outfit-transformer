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

import faiss
from tqdm import tqdm
import pathlib

from ..utils import utils


def faiss_exists(index_path):
    if os.path.exists(index_path):
        return True
    else:
        return False


def create_faiss(
    faiss_type, 
    d_embed, 
    *faiss_args, **faiss_kwargs
):
    if faiss_type == 'IndexFlatIP':
        index = faiss.IndexFlatIP(
            d_embed, 
            *faiss_args, **faiss_kwargs
        )
    elif faiss_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(
            d_embed, 
            *faiss_args, **faiss_kwargs
        )
    else:
        raise ValueError("Invalid FAISS index type")
    
    index = faiss.IndexIDMap2(index)
    print("[FAISS] created")
    
    return index


def add(
    index: faiss.Index, 
    embeddings: List[List[float]], 
    ids: List[int],
    batch_size: int = 2048,
):
    iterable = tuple(zip(embeddings, ids))
    for batch in utils.batch_iterable(iterable, batch_size, desc="[FAISS] Adding"):
        embeddings, ids = zip(*batch)
        index.add_with_ids(np.array(embeddings), np.array(ids))


def search(
    index: faiss.Index, 
    embeddings: List[List[float]], 
    k: int,
    batch_size: int = 2048,
) -> List[Tuple[float, int]]:
    outputs = []
    for batch in utils.batch_iterable(embeddings, batch_size, desc="[FAISS] Searching"):
        scores, faiss_ids = index.search(
            np.array(batch), k=k
        )
        scores = scores.tolist()
        faiss_ids = faiss_ids.tolist()
        for scores_, faiss_ids_ in zip(scores, faiss_ids):
            outputs.append(list(zip(scores_, faiss_ids_)))
        
    return outputs


def save(
    index: faiss.Index, 
    index_path: str
):
    faiss.write_index(index, index_path)
    print("[FAISS] saved")