import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Literal
from PIL import Image
import torch
import random
from argparse import ArgumentParser

from .pipeline import BasePipeline
from ..data import datatypes

ITEM_PER_PAGE = 12


my_items = []

candidate_items = []

POLYVORE_CATEGORIES = [
    'all-body', 'bottoms', 'tops', 'outerwear', 'bags', 
    'shoes', 'accessories', 'scarves', 'hats', 
    'sunglasses', 'jewellery', 'unknown'
]


def run(
    pipeline: BasePipeline,
    task: Literal['cp', 'cir']
):
    with gr.Blocks() as demo:
        outfit_selected_idx = gr.State(value=None)
        
        gr.Markdown(
            "## Compatibility Prediction"
        )
        
        gr.Markdown(
            "### 1. Add Items"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                input_category = gr.Dropdown(
                    label="Category",
                    choices=POLYVORE_CATEGORIES,
                    value=None,
                )
                input_description = gr.Textbox(
                    label="Enter Description",
                )
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                )
                
            with gr.Column(scale=8, variant='panel'):
                candidates_gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=4,
                    rows=3,
                    type="pil",
                )
                candidates_gallery_page = gr.Dropdown(
                    choices=[1], value=1,
                    label="Page",
                )


            def __input_category_change(category):
                global candidate_items
                candidate_items = pipeline.loader.paginate(
                    item_per_page=ITEM_PER_PAGE, 
                    page=1, 
                    category=category
                )
                max_page = pipeline.loader.total_pages(
                    item_per_page=ITEM_PER_PAGE, 
                    category=category
                )
                candidates_gallery_page = gr.Dropdown(
                    choices=list(range(1, max_page + 1)), value=1,
                    label="Page",
                )
                return [item.image for item in candidate_items],candidates_gallery_page
                # 
            input_category.change(
                __input_category_change,
                inputs=[input_category],
                outputs=[candidates_gallery, candidates_gallery_page]
            )
            
            
            def __candidates_gallery_page_change(page, category):
                global candidate_items
                candidate_items = pipeline.loader.paginate(
                    item_per_page=ITEM_PER_PAGE, 
                    page=page, 
                    category=category
                )
                return [item.image for item in candidate_items]
            
            candidates_gallery_page.change(
                __candidates_gallery_page_change,
                inputs=[candidates_gallery_page, input_category],
                outputs=[candidates_gallery]
            )
            
            
            def __candidates_gallery_select(evt: gr.SelectData):
                selected_item = candidate_items[evt.index]
                return selected_item.image, selected_item.description
                
            candidates_gallery.select(
                __candidates_gallery_select,
                inputs=None,
                outputs=[input_image, input_description]
            )
            
            
        with gr.Row():
            btn_add_my_item = gr.Button("Add Item to Outfit")
            
        gr.Markdown(
            "### 2. Check Outfit"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=8, variant='panel'):
                inputs_gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=6,
                    rows=1,
                )
                with gr.Row():
                    btn_inputs_gallery_clear = gr.Button(
                        "Clear All Items",
                    )
                    btn_inputs_gallery_delete = gr.Button(
                        "Delete Item",
                    )
                    
                    
        def __btn_add_my_item_click(img, desc, cat):
            my_items.append(
                datatypes.FashionItem(
                    id=str(random.randint(0, 1000)),
                    image=img, 
                    description=desc,
                    category=cat,
                )
            )
            return [item.image for item in my_items], None
            
        btn_add_my_item.click(
            __btn_add_my_item_click,
            inputs=[input_image, input_description, input_category],
            outputs=[inputs_gallery, outfit_selected_idx]
        )
        
        
        def __inputs_gallery_select(selected: gr.SelectData):
            return selected.index
        
        inputs_gallery.select(
            __inputs_gallery_select,
            inputs=None,
            outputs=outfit_selected_idx
        )
        
        
        def __btn_inputs_gallery_clear_click():
            global my_items
            my_items = []
            
            return None, None

        btn_inputs_gallery_clear.click(
            __btn_inputs_gallery_clear_click,
            inputs=None,
            outputs=[outfit_selected_idx, inputs_gallery]
        )
        
        
        def __btn_inputs_gallery_delete_click(idx):
            my_items.pop(idx)
            return [item.image for item in my_items], None
            
        btn_inputs_gallery_delete.click(
            __btn_inputs_gallery_delete_click,
            inputs=outfit_selected_idx,
            outputs=[inputs_gallery, outfit_selected_idx]
        )
        
        
        if task == 'cp':   
            gr.Markdown(
                "### 3. Compute Score"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, variant='panel'):
                    btn_evaluate = gr.Button(
                        "Evaluate",
                        variant="primary"
                    )
                with gr.Column(scale=8, variant='panel'):
                    score = gr.Textbox(
                        label="Compatibility Score",
                        interactive=False
                    )
            
            
            def __btn_evaluate_click():
                score_ = pipeline.compatibility_predict(
                    query=datatypes.FashionCompatibilityQuery(outfit=my_items)
                )
                return score_
            btn_evaluate.click(
                __btn_evaluate_click,
                inputs=None,
                outputs=score
            )
                    
                    
        elif task == 'cir':
            gr.Markdown(
                "### 3. Search"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, variant='panel'):
                    category = gr.Radio(
                        label="Category",
                        choices=constants.POLYVORE_CATEGORIES,
                        value="",
                    )
                    btn_search = gr.Button(
                        "Search",
                        variant="primary"
                    )
                with gr.Column(scale=8, variant='panel'):
                    search_result_gallery = gr.Gallery(
                        allow_preview=False,
                        show_label=True,
                        columns=4,
                        rows=3,
                        type="pil",
                    )
                    
            def __btn_search_click(category):
                items = pipeline.complementary_search(
                    query=datatypes.FashionComplementaryQuery(category=category, outfit=my_items),
                    k=ITEM_PER_PAGE
                )
                return [item.image for item in items]
            
            btn_search.click(
                __btn_search_click,
                inputs=category,
                outputs=search_result_gallery
            )

    demo.launch()
