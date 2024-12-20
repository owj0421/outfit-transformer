# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import datetime
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle

from PIL import Image
import torchvision.transforms as transforms
import json
import random

from ..utils.elements import Item, Outfit, Query
        
        
class PolyvoreItems():
    
    def __init__(
        self,
        dataset_dir,
    ):
        self.items = []
        self.item_id_to_idx = {}
        self.item_id_by_category = {}
        
        with open(os.path.join(dataset_dir, 'item', 'metadata.json'), 'r') as f:
            metadatas = json.load(f)
            
        for item_id, item_ in metadatas.items():
            self.item_id_to_idx[item_id] = len(self.items)
            self.items.append(
                {
                    'id': item_id,
                    'category': item_['semantic_category'],
                    'image_path': os.path.join(dataset_dir, 'images', f"{item_id}.jpg"),
                    'description': item_['description'] if item_['description'] else item_['url_name'],
                }
            )
            if item_['semantic_category'] not in self.item_id_by_category:
                self.item_id_by_category[item_['semantic_category']] = []
            self.item_id_by_category[item_['semantic_category']].append(item_id)
        
    def __len__(self):
        return len(self.items)
    
    def __call__(self, item_id):
        idx = self.item_id_to_idx[item_id]
        item = self.items[idx]
        item = Item(
            id=item['id'],
            category=item['category'],
            image=Image.open(item['image_path']),
            description=item['description']
        )
        
        return item
    
    def sample_by_category(self, n_samples, category: str=None):
        if category is None:
            item_ids = list(self.item_id_to_idx.keys())
        else:
            item_ids = self.item_id_by_category[category]
        
        return random.sample(item_ids, n_samples)
    
    def get_category(self, item_id):
        idx = self.item_id_to_idx[item_id]
        
        return self.items[idx]['category']
        

class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        polyvore_type: Literal[
            'nondisjoint',
            'disjoint',
        ] = 'nondisjoint',
        split: Literal[
            'train',
            'valid',
            'test',
        ] = 'train',
    ):
        path = os.path.join(
            dataset_dir, polyvore_type, 'compatibility', f"{split}.json"
        )
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.polyvore_type = polyvore_type
        self.split = split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        outfit = self.data[idx]
        
        return outfit
    
    def collate_fn(self, batch):
        label = [float(item['label']) for item in batch]
        question = [item['question'] for item in batch]

        return {
            'label': label,
            'question': question
        }
        
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        polyvore_items: PolyvoreItems,
        dataset_dir: str,
        polyvore_type: Literal[
            'nondisjoint',
            'disjoint',
        ] = 'nondisjoint',
        split: Literal[
            'train',
            'valid',
            'test',
        ] = 'train',
    ):
        self.polyvore_items = polyvore_items
        path = os.path.join(
            dataset_dir, polyvore_type, 'fill_in_the_blank', f"{split}.json"
        )
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.polyvore_type = polyvore_type
        self.split = split
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        outfit = self.data[idx]
        outfit['candidates'] = outfit['answers']
        outfit['category'] = self.polyvore_items.get_category(outfit['candidates'][int(outfit['label'])])
        
        return outfit
    
    def collate_fn(self, batch):
        label = [item['label'] for item in batch]
        question = [item['question'] for item in batch]
        candidates = [item['candidates'] for item in batch]
        blank_positions = [item['blank_position'] for item in batch]
        category = [item['category'] for item in batch]

        return {
            'label': label,
            'question': question,
            'candidates': candidates,
            'blank_position': blank_positions,
            'category': category
        }
        
        
class PolyvoreTripletDataset(Dataset):
    
        def __init__(
            self,
            polyvore_items: PolyvoreItems,
            dataset_dir: str,
            polyvore_type: Literal[
                'nondisjoint',
                'disjoint',
            ] = 'nondisjoint',
            split: Literal[
                'train',
                'valid',
                'test',
            ] = 'train',
            sampling_strategy: Literal[
                'in-batch', 'all', 'same_category'
            ] = 'same_category',
            n_samples: Optional[Dict[Literal['all', 'hard'], int]] = {'hard': 2, 'all': 8},
        ):
            self.polyvore_items = polyvore_items
            path = os.path.join(
                dataset_dir, polyvore_type, f"{split}.json"
            )
            with open(path, 'r') as f:
                self.data = json.load(f)
            self.polyvore_type = polyvore_type
            self.split = split
            
            self.set_id = list(self.data.keys())
            self.sampling_strategy = sampling_strategy
            self.n_samples = n_samples
            
        def __len__(self):
            return len(self.set_id)
        
        def __getitem__(self, idx):
            set_id = self.set_id[idx]
            item_ids = self.data[set_id]['item_ids']
            
            random.shuffle(item_ids)
            
            anchor = item_ids[:-1]
            positive = [item_ids[-1]]
            
            category = self.polyvore_items.get_category(positive[0])
            
            all_negative = self.polyvore_items.sample_by_category(n_samples=self.n_samples['all'])
            hard_negative = self.polyvore_items.sample_by_category(n_samples=self.n_samples['hard'], category=category)
            
            return {
                "anchor": anchor,
                "positive": positive,
                "all_negative": all_negative,
                "hard_negative": hard_negative,
                "category": category
            }
        
        def collate_fn(self, batch):
            anchor = [item['anchor'] for item in batch]
            positive = [item['positive'] for item in batch]
            all_negative = [item['all_negative'] for item in batch]
            hard_negative = [item['hard_negative'] for item in batch]
            category = [item['category'] for item in batch]
            
            return {
                'anchor': anchor,
                'positive': positive,
                'all_negative': all_negative,
                'hard_negative': hard_negative,
                'category': category
            }
            
            
if __name__ == '__main__':
    polyvore_items = PolyvoreItems(
        dataset_dir='/home/owj0421/datasets/polyvore'
    )
    print(polyvore_items.items[0])
    
    print(polyvore_items.item_id_by_category.keys())
    
    # polyvore_triplet_dataset = PolyvoreTripletDataset(
    #     polyvore_items=polyvore_items,
    #     dataset_dir='/home/owj0421/datasets/polyvore',
    #     polyvore_type='nondisjoint',
    #     split='train',
    #     sampling_strategy='same_category',
    #     n_samples=4
    # )
    # polyvore_fitb_dataset = PolyvoreFillInTheBlankDataset(
    #     polyvore_items=polyvore_items,
    #     dataset_dir='/home/owj0421/datasets/polyvore',
    #     polyvore_type='nondisjoint',
    #     split='train'
    # )
    
    # print(polyvore_triplet_dataset[0])
    # print(polyvore_fitb_dataset[0])