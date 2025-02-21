# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
import random

from ..datatypes import (
    FashionItem, 
    FashionCompatibilityQuery, 
    FashionComplementaryQuery, 
    FashionCompatibilityData, 
    FashionFillInTheBlankData, 
    FashionTripletData
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
        
    return metadata


def load_item(dataset_dir, metadata, item_id):
    item = metadata[item_id]
    
    image_path = POLYVORE_IMAGE_DATA_PATH.format(
        dataset_dir=dataset_dir,
        item_id=item_id
    )
    image = Image.open(image_path).convert('RGB')
    
    return FashionItem(
        item_id=item['item_id'],
        category=item['semantic_category'],
        image=image,
        description=item['title'] if item['title'] else item['url_name'],
        metadata=item
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
    ):
        self.dataset_dir = dataset_dir
        self.metadata = load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'compatibility', dataset_split
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionCompatibilityData:
        label = self.data[idx]['label']

        outfit = [
            load_item(self.dataset_dir, self.metadata, item_id) 
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
    ):
        self.dataset_dir = dataset_dir
        self.metadata = load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'fill_in_the_blank', dataset_split
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionFillInTheBlankData:
        label = self.data[idx]['label']
        answers = [
            load_item(self.dataset_dir, self.metadata, item_id) 
            for item_id in self.data[idx]['answers']
        ]
        query = FashionComplementaryQuery(
            outfit=[
                load_item(self.dataset_dir, self.metadata, item_id) 
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
    ):
        self.dataset_dir = dataset_dir
        self.metadata = load_metadata(dataset_dir)
        self.data = load_set_data(
            dataset_dir, dataset_type, dataset_split
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionTripletData:
        items = [
            load_item(self.dataset_dir, self.metadata, item_id) 
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