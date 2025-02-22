import os
import sys
import json
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from ..data import datatypes
from ..data.datasets.polyvore import POLYVORE_METADATA_PATH
from ..demo.stores.metadata import ItemMetadataStore


import pathlib


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
LOADER_DIR = SRC_DIR / 'stores'
RESULT_DIR = SRC_DIR / 'results'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--polyvore_dir', 
        type=str, required=True
    )
    return parser.parse_args()


def main(args):
    print(
        "Building database from Polyvore"
    )
    loader = ItemMetadataStore(
        database_name='polyvore',
        table_name='items',
        base_dir=LOADER_DIR
    )
    
    path = POLYVORE_METADATA_PATH.format(
        dataset_dir=args.polyvore_dir
    )
    with open(path, 'r') as f:
        metadatas = json.load(f)
        
    items = []
    
    for item in tqdm(metadatas):
        image_path=os.path.join(args.polyvore_dir, 'images', str(item['item_id']) + '.jpg')
        
        # Load Image PIL
        image = Image.open(image_path)
        image = image.convert('RGB')
            
        items.append(datatypes.FashionItem(
            item_id=item['item_id'],
            category=item['semantic_category'].lower().strip(),
            image=image,
            description=item['url_name'], 
            metadata={}
        ))
        
        if len(items) == 1000:
            loader.add(items)
            items = []
    
    if items:
        loader.add(items)
        items = []
            
    print(
        "Database built."
    )
    
if __name__ == "__main__":
    args = parse_args()
    main(args)