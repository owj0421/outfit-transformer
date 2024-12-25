import torch
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitClipTransformerConfig,
    OutfitClipTransformer
)


def load_model(model_type, checkpoint):
    if model_type == 'original':
        cfg = OutfitTransformerConfig(
            nhead=16, #16
            num_layers=6, #6
            dim_feedforward=2048,
            dropout=0.3,
        )
        model = OutfitTransformer(
            cfg
        ).cuda()

    elif model_type == 'clip':
        cfg = OutfitClipTransformerConfig(
            nhead=16,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.3,
        )
        model = OutfitClipTransformer(
            cfg
        ).cuda()
        
    if checkpoint:
        model.load_state_dict(
            torch.load(
                checkpoint
            ),
            strict=False
        )
        print(
            f"Loaded model from checkpoint: {checkpoint}"
        )
            
    return model