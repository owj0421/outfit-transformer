import torch
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitCLIPTransformerConfig,
    OutfitCLIPTransformer
)


def load_model(model_type, checkpoint):
    if model_type == 'original':
        cfg = OutfitTransformerConfig()
        model = OutfitTransformer(
            cfg
        ).cuda()

    elif model_type == 'clip':
        cfg = OutfitCLIPTransformerConfig()
        model = OutfitCLIPTransformer(
            cfg
        ).cuda()
        
    if checkpoint:
        model.load_state_dict(
            torch.load(checkpoint),
            strict=False
        )
        print(
            f"Loaded model from checkpoint: {checkpoint}"
        )
            
    return model