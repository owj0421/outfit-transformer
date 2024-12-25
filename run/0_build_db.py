import sys
from fashion_recommenders.fashion_recommenders.data.loader import SQLiteItemLoader
from fashion_recommenders.fashion_recommenders.utils.elements import Item

import json
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--polyvore_dir', 
        type=str, required=True
    )
    parser.add_argument(
        "--db_dir", type=str, default="./src/db",
        help="dir path to save index"
    )
    return parser.parse_args()

def main(args):
    print(
        "Building database from Polyvore"
    )
    loader = SQLiteItemLoader(
        db_dir=args.db_dir,
        # image_dir=os.path.join(args.polyvore_dir, 'images'),
    )

    with open(os.path.join(args.polyvore_dir, 'item', 'metadata.json'), 'r') as f:
        metadatas = json.load(f)
        
    items = [
        Item(
            item_id=int(item_id),
            image=None,
            image_path=os.path.join(args.polyvore_dir, 'images', str(item_id) + '.jpg'),
            description=item_['title'] if item_['title'] else item_['url_name'], 
            category=item_['semantic_category'].lower().strip()
        ) for item_id, item_ in metadatas.items()
    ]
    loader.add(items)

    del loader
    
    print(
        "Database built."
    )
    
if __name__ == "__main__":
    args = parse_args()
    main(args)