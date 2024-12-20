import gradio as gr
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import torch
import random
from src.model.load import load_model
from src.utils.elements import Item, Outfit, Query
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type', 
        type=str,  choices=['original', 'clip'], default='clip'
    )
    parser.add_argument(
        '--polyvore_dir', 
        type=str, default='../datasets/polyvore'
    )
    parser.add_argument(
        '--task', 
        type=str, choices=['cp', 'cir'], default='cp'
    )
    parser.add_argument(
        '--checkpoint',
        type=str, default='./checkpoints/cp-dainty-voice-70/epoch_5_acc_0.859_loss_0.035/model.pt'
    )
    parser.add_argument(
        '--index_dir',
        type=str, default='./index'
    )
    return parser.parse_args()


class OutfitManager:
    def __init__(self, model):
        self.outfit_items = []
        self.model = model
        model.eval()

    def add_item(self, image, description):
        if image and description:
            new_item = Item(category=None, image=image, description=description)
            self.outfit_items.append(new_item)
            return "Item added!", self.get_item_images()
        else:
            return (
                "Please upload an image and provide a description.", 
                self.get_item_images()
            )

    def delete_item(self, idx):
        try:
            if 0 <= idx < len(self.outfit_items):
                self.outfit_items.pop(idx)
                return (
                    f"Item {idx} deleted!", 
                    self.get_item_images()
                )
            else:
                return (
                    "Invalid item index.", 
                    self.get_item_images()
                )
        except:
            return (
                "Invalid item index.", 
                self.get_item_images()
            )

    def clear_items(self):
        self.outfit_items = []
        return (
            "All items cleared!", 
            self.get_item_images()
        )

    def get_item_images(self):
        images = (
            [item.image for item in self.outfit_items] 
            if self.outfit_items else None
        )
        return images

    @torch.no_grad()
    def evaluate_outfit(self):
        outfit = Outfit(
            items=self.outfit_items
        )
        score = float(model.predict(
            [outfit]
        )[0])
        return score
    
    @torch.no_grad()
    def encode_query(self, query: str):
        inputs = [Query(
            query=query,
            items=self.outfit_items
        )]
        return model.embed_query(
            inputs
        )[0].detach().cpu().numpy()


def get_item_idx(
    selected: gr.SelectData
):
    return selected.index


def build_cp_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            "## CIR Demo"
        )
        gr.Markdown(
            "### Add your outfit items and query, then find most relevant items!"
        )
        msg = gr.Markdown(
            "Messages will appear here."
        )
        selected_idx = gr.State(
            value=None
        )
        with gr.Row():
            with gr.Column(scale=2, variant='panel'):
                image = gr.Image(
                    label="Image",
                    type="pil",
                    height=240,
                )
                description = gr.Textbox(
                    label="Description",
                )
                btn_add = gr.Button(
                    "Add Item",
                )  
            with gr.Column(scale=8, variant='panel'):
                gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=8,
                    height=345,
                )# .style(grid=[1], container=True, scrollable=True)
                with gr.Row():
                    btn_clear_items = gr.Button(
                        "Clear All Items",
                    )
                    btn_delete_item = gr.Button(
                        "Delete Item",
                    )
            with gr.Column(scale=2):
                with gr.Row():
                    score = gr.Textbox(
                        label="Compatibility Score",
                        interactive=False
                    )
                    btn_evaluate = gr.Button(
                        "Evaluate",
                        variant="primary"
                    )

        btn_add.click(
            manager.add_item, 
            inputs=[image, description], 
            outputs=[msg, gallery]
        )
        btn_clear_items.click(
            manager.clear_items, 
            outputs=[msg, gallery]
        )
        btn_delete_item.click(
            manager.delete_item, 
            inputs=selected_idx, 
            outputs=[msg, gallery]
        )
        btn_evaluate.click(
            manager.evaluate_outfit, 
            outputs=score
        )
        gallery.select(
            get_item_idx, 
            inputs=None, 
            outputs=selected_idx
        )

    return demo



def search_items(query: str):
    embeddings = manager.encode_query(query)
    res = indexer.search(
        query_embeddings=embeddings,
        top_k=8
    )[0]
    images = [
        polyvore_items(i['id']).image 
        for i in res
    ]
    
    return images


def build_cir_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            "## Outfit Compatibility Demo"
        )
        gr.Markdown(
            "### Add your outfit items and see how well they match!"
        )
        msg = gr.Markdown(
            "Messages will appear here."
        )
        selected_idx = gr.State(
            value=None
        )
        with gr.Row():
            result_gallery = gr.Gallery(
                allow_preview=False,
                show_label=True,
                columns=8,
                height=345,
            )# .style(grid=[1], container=True, scrollable=True)
        
        with gr.Row():
            with gr.Column(scale=2, variant='panel'):
                image = gr.Image(
                    label="Image",
                    type="pil",
                    height=240,
                )
                description = gr.Textbox(
                    label="Description",
                )
                btn_add = gr.Button(
                    "Add Item",
                )  
            with gr.Column(scale=8, variant='panel'):
                gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=8,
                    height=345,
                )# .style(grid=[1], container=True, scrollable=True)
                with gr.Row():
                    btn_clear_items = gr.Button(
                        "Clear All Items",
                    )
                    btn_delete_item = gr.Button(
                        "Delete Item",
                    )
            with gr.Column(scale=2):
                with gr.Row():
                    category = gr.Radio(
                        label="Category",
                        choices=[
                            'tops', 'bottoms', 'jewellery', 'shoes', 'outerwear', 
                            'scarves', 'sunglasses', 'bags', 'hats', 'all-body', 'accessories'
                        ]
                    )
                    btn_search = gr.Button(
                        "Search",
                        variant="primary"
                    )


        btn_add.click(
            manager.add_item, 
            inputs=[image, description], 
            outputs=[msg, gallery]
        )
        btn_clear_items.click(
            manager.clear_items, 
            outputs=[msg, gallery]
        )
        btn_delete_item.click(
            manager.delete_item, 
            inputs=selected_idx, 
            outputs=[msg, gallery]
        )
        btn_search.click(
            search_items, 
            inputs=category,
            outputs=result_gallery
        )
        gallery.select(
            get_item_idx, 
            inputs=None, 
            outputs=selected_idx
        )

    return demo


if __name__ == "__main__":
    args = parse_args()
    model = load_model(
        args
    )
    manager = OutfitManager(
        model=model
    )
    if args.task == 'cp':
        demo = build_cp_ui()
    else:
        from src.index.indexer import Indexer
        indexer = Indexer.load_local(
            args.index_dir
        )
        from src.datasets.polyvore import PolyvoreItems
        polyvore_items = PolyvoreItems(
            args.polyvore_dir, 
        )
        demo = build_cir_ui()
        
    demo.launch()
