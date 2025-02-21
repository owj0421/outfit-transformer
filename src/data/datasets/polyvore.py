# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
import os
import cv2
import json
import random
import pickle
from tqdm import tqdm
from ..datatypes import (
    FashionItem, 
    FashionCompatibilityQuery, 
    FashionComplementaryQuery, 
    FashionCompatibilityData, 
    FashionFillInTheBlankData, 
    FashionTripletData
)
from functools import lru_cache
import numpy as np

POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = (
    "{dataset_dir}/precomputed_clip_embeddings"
)
POLYVORE_METADATA_PATH = (
    "{dataset_dir}/item_metadata.json"
)
POLYVORE_SET_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_split}.json"
)
POLYVORE_TASK_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_task}/{dataset_split}.json"
)
POLYVORE_IMAGE_DATA_PATH = (
    "{dataset_dir}/images/{item_id}.jpg"
)

def load_metadata(dataset_dir):
    metadata = {}
    with open(
        POLYVORE_METADATA_PATH.format(dataset_dir=dataset_dir), 'r'
    ) as f:
        metadata_ = json.load(f)
        for item in metadata_:
            metadata[item['item_id']] = item
    print(f"Loaded {len(metadata)} metadata")
    return metadata


def load_all_embeddings_dict(dataset_dir):
    e_dir = POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR.format(dataset_dir=dataset_dir)
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


def load_image(dataset_dir, item_id, size=(224, 224)):
    image_path = POLYVORE_IMAGE_DATA_PATH.format(
        dataset_dir=dataset_dir,
        item_id=item_id
    )
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)  # Lanczos 대신 Bilinear
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_image_wrapper(args):
    dataset_dir, item_id, size = args
    return item_id, load_image(dataset_dir, item_id, size)
    
    
def load_all_image_dict(dataset_dir, metadata, size=(224, 224)):
    all_image_dict = {}
    num_workers = min(cpu_count(), 8)  # 최대 8개 프로세스 사용

    with Pool(num_workers) as p:
        results = list(tqdm(
            p.imap(load_image_wrapper, [(dataset_dir, item_id, size) for item_id in metadata.keys()]),
            total=len(metadata),
            desc="Loading Images"
        ))

    all_image_dict = {item_id: img for item_id, img in results if img is not None}
    print(f"Loaded {len(all_image_dict)} images")
    return all_image_dict


def load_item(dataset_dir, metadata, item_id, all_embeddings_dict=None, all_image_dict=None):
    metadata_ = metadata[item_id]
    
    if all_embeddings_dict:
        embedding = all_embeddings_dict[item_id]
    else:
        embedding = None
        
    if all_image_dict:
        image = all_image_dict[item_id]
    else:
        image = load_image(dataset_dir, item_id) if not all_embeddings_dict else None
    
    return FashionItem(
        item_id=metadata_['item_id'],
        category=metadata_['semantic_category'],
        image=image,
        description=metadata_['title'] if metadata_['title'] else metadata_['url_name'],
        metadata=metadata_,
        embedding=embedding
    )
    
    
def load_task_data(dataset_dir, dataset_type, task, dataset_split):
    with open(
        POLYVORE_TASK_DATA_PATH.format(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            dataset_task=task,
            dataset_split=dataset_split
        ), 'r'
    ) as f:
        data = json.load(f)
        
    return data


def load_set_data(dataset_dir, dataset_type, dataset_split):
    with open(
        POLYVORE_SET_DATA_PATH.format(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            dataset_split=dataset_split
        ), 'r'
    ) as f:
        data = json.load(f)
        
    return data


class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        all_embeddings_dict=None,
        all_image_dict=None
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'compatibility', dataset_split
        )
        self.all_embeddings_dict = all_embeddings_dict
        self.all_image_dict = all_image_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionCompatibilityData:
        label = self.data[idx]['label']

        outfit = [
            load_item(self.dataset_dir, self.metadata, item_id, self.all_embeddings_dict) 
            for item_id in self.data[idx]['question']
        ]
        query=FashionCompatibilityQuery(
            outfit=outfit
        )
        
        return FashionCompatibilityData(
            label=label,
            query=query
        )
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        all_embeddings_dict=None,
        all_image_dict=None
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'fill_in_the_blank', dataset_split
        )
        self.all_embeddings_dict = all_embeddings_dict
        self.all_image_dict = all_image_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionFillInTheBlankData:
        label = self.data[idx]['label']
        answers = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.all_embeddings_dict, self.all_image_dict) 
            for item_id in self.data[idx]['answers']
        ]
        query = FashionComplementaryQuery(
            outfit=[
                load_item(self.dataset_dir, self.metadata, item_id, 
                          self.all_embeddings_dict, self.all_image_dict) 
                for item_id in self.data[idx]['question']
            ],
            category=answers[label].category
        )

        return FashionFillInTheBlankData(
            query=query,
            label=label,
            answers=answers
        )
    
        
class PolyvoreTripletDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        all_embeddings_dict=None,
        all_image_dict=None
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_set_data(
            dataset_dir, dataset_type, dataset_split
        )
        self.all_embeddings_dict = all_embeddings_dict
        self.all_image_dict = all_image_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionTripletData:
        items = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.all_embeddings_dict, self.all_image_dict) 
            for item_id in self.data[idx]['item_ids']
        ]
        answer = items[random.randint(0, len(items) - 1)]
        outfit = [item for item in items if item != answer]
        query = FashionComplementaryQuery(
            outfit=outfit,
            category=answer.category
        )
        return FashionTripletData(
            query=query,
            answer=answer
        )
        
        
class PolyvoreItemDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        metadata: dict = None,
        all_embeddings_dict=None,
        all_image_dict=None
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.all_item_ids = list(self.metadata.keys())
        self.all_embeddings_dict = all_embeddings_dict
        self.all_image_dict = all_image_dict
        
    def __len__(self):
        return len(self.all_item_ids)
    
    def __getitem__(self, idx) -> FashionItem:
        item = load_item(self.dataset_dir, self.metadata, self.all_item_ids[idx], 
                         self.all_embeddings_dict, self.all_image_dict)

        return item
        
        
if __name__ == '__main__':
    # Test the dataset
    dataset_dir = "/home/owj0421/datasets/polyvore"
    
    dataset = PolyvoreCompatibilityDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreFillInTheBlankDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreTripletDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])