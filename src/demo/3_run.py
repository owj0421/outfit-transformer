import os
import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Literal
from PIL import Image
import torch
import random
from argparse import ArgumentParser
import pathlib

from .vectorstore import FAISSVectorStore
from ..models.load import load_model
from ..data import datatypes
from ..data.datasets import polyvore

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"

ITEM_PER_PAGE = 12
ITEM_PER_SEARCH = 8

POLYVORE_CATEGORIES = [
    'all-body', 'bottoms', 'tops', 'outerwear', 'bags', 
    'shoes', 'accessories', 'scarves', 'hats', 
    'sunglasses', 'jewellery', 'unknown'
]
state_my_items = []
state_candidate_items = []


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    
    return parser.parse_args()


def run(args):
    
    metadata = polyvore.load_metadata(
        args.polyvore_dir
    )
    items = polyvore.PolyvoreItemDataset(
        args.polyvore_dir, metadata=metadata, load_image=True
    )
    num_pages = len(items) // ITEM_PER_PAGE
    
    def get_items(page):
        idxs = range(page * ITEM_PER_PAGE, (page + 1) * ITEM_PER_PAGE)
        return [items[i] for i in idxs]

    
    model = load_model(
        model_type=args.model_type, checkpoint=args.checkpoint
    )
    model.eval()
    indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir),
    )

    with gr.Blocks() as demo:
        state_selected_my_item_index = gr.State(value=None)
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "# Outfit Recommendation Demo"
            )
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## My Items"
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=10, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        my_item_gallery = gr.Gallery(
                            allow_preview=False, show_label=True,
                            columns=4, rows=1,
                        )       
            with gr.Column(scale=2, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_category = gr.Dropdown(
                            label="Category",
                            choices=POLYVORE_CATEGORIES, value=None,
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_image = gr.Image(
                            label="Upload Image", type="pil",
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_description = gr.Textbox(
                            label="Description",
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=100, variant='compact'):
                        btn_item_add = gr.Button("Add")
                    with gr.Column(scale=1, min_width=100, variant='compact'):
                        btn_item_delete = gr.Button("Delete")
        
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## Add From Polyvore"
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=12, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        polyvore_gallery = gr.Gallery(
                            allow_preview=False,
                            show_label=True,
                            columns=[ITEM_PER_PAGE // 2], 
                            rows=[2],
                            type="pil",
                            object_fit='contain',
                            height='auto'
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        polyvore_page = gr.Dropdown(
                            label="Page",
                            choices=list(range(1, num_pages + 1)),  # 1부터 num_pages까지 선택 가능
                            value=None  # 기본값
                        )
        
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## Task"
            )
        with gr.Row(equal_height=True, variant='compact'):
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "Compute Score"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    btn_compute_score = gr.Button(
                        "Compute",
                        variant="primary"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    computed_score = gr.Textbox(
                        label="Compatibility Score",
                        interactive=False
                    )
                        
            with gr.Column(scale=2, variant='compact'):
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "Search Complementary Items"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    btn_search_item = gr.Button(
                        "Search", variant="primary"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    searched_item_gallery = gr.Gallery(
                        allow_preview=False,
                        show_label=True,
                        columns=[ITEM_PER_SEARCH // 2], 
                        rows=[2],
                        type="pil",
                        object_fit='contain',
                        height='auto'
                    )
                    
        # Functions
        def select_item(selected: gr.SelectData):

            return {
                state_selected_my_item_index: selected.index,
                item_image: state_my_items[selected.index].image,
                item_description: state_my_items[selected.index].description,
                item_category: state_my_items[selected.index].category,
            }
        
        def add_item(item_image, item_description, item_category):
            global state_my_items
            
            if item_image is None or item_description is None or item_category is None:
                gr.Warning("Error: All fields (image, description, and category) must be provided.")
            else:
                state_my_items.append(
                    datatypes.FashionItem(
                        id=None,
                        image=item_image, 
                        description=item_description,
                        category=item_category,
                    )
                )
                
            return {
                my_item_gallery: [item.image for item in state_my_items],
                state_selected_my_item_index: None,
            }
        
        def delete_item(index):
            if index is not None:
                if index < len(state_my_items):
                    del state_my_items[index]
                else:
                    gr.Warning("Error: Invalid item index.")
            else:
                gr.Warning("Error: No item selected.")
            
            return {
                my_item_gallery: [item.image for item in state_my_items],
                state_selected_my_item_index: None,
                item_image: None,
                item_description: None,
                item_category: None,
            }
        
        def select_page_from_polyvore(page):
            global state_candidate_items
            
            page = page - 1
            state_candidate_items = get_items(page)
            
            return {
                polyvore_gallery: [item.image for item in state_candidate_items],
            }
        
        def select_item_from_polyvore(selected: gr.SelectData):
            selected_item = state_candidate_items[selected.index]
            
            return {
                item_image: selected_item.image,
                item_description: selected_item.description,
                item_category: selected_item.category,
            }
        
        @torch.no_grad()
        def compute_score():
            if len(state_my_items) == 0:
                gr.Warning("Error: No items to compute score.")
                return {
                    computed_score: None
                }
            query = datatypes.FashionCompatibilityQuery(
                outfit=state_my_items
            )
            s = model.predict_score(
                query= [query],
                use_precomputed_embedding=False
            )[0].detach().cpu()
            s = float(s)
            
            return {
                computed_score: s
            }
        
        @torch.no_grad()
        def search_item():
            if len(state_my_items) == 0:
                gr.Warning("Error: No items to search.")
                return {
                    searched_item_gallery: []
                }
            query = datatypes.FashionComplementaryQuery(
                outfit=state_my_items,
                category='Unknown'
            )
            
            e = model.embed_query(
                query=[query],
                use_precomputed_embedding=False
            ).detach().cpu().numpy().tolist()
            
            res = indexer.search(
                embeddings=e,
                k=ITEM_PER_SEARCH
            )[0]
            
            return {
                searched_item_gallery: [items.get_item_by_id(r[1]).image for r in res]
            }
            
        
        # Event Handlers
        my_item_gallery.select(
            select_item,
            inputs=None,
            outputs=[state_selected_my_item_index, item_image, item_description, item_category]
        )
        btn_item_add.click(
            add_item, 
            inputs=[item_image, item_description, item_category], 
            outputs=[my_item_gallery, state_selected_my_item_index]
        )
        btn_item_delete.click(
            delete_item,
            inputs=state_selected_my_item_index,
            outputs=[my_item_gallery, state_selected_my_item_index, item_image, item_description, item_category]
        )
        
        polyvore_page.change(
            select_page_from_polyvore,
            inputs=[polyvore_page],
            outputs=[polyvore_gallery]
        )
        polyvore_gallery.select(
            select_item_from_polyvore,
            inputs=None,
            outputs=[item_image, item_description, item_category]
        )
        
        btn_compute_score.click(
            compute_score,
            inputs=None,
            outputs=[computed_score]
        )
        btn_search_item.click(
            search_item,
            inputs=None,
            outputs=[searched_item_gallery]
        )
    
    # Launch
    demo.launch()
    
if __name__ == "__main__":
    args = parse_args()
    run(args)