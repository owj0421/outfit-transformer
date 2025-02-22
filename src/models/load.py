import torch
from typing import Any, Dict, Optional
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitCLIPTransformerConfig,
    OutfitCLIPTransformer
)


def load_model(model_type, checkpoint, **cfg_kwargs):
    if checkpoint:
        state_dict = torch.load(checkpoint)
        
    if model_type == 'original':
        cfg = OutfitTransformerConfig(**state_dict['cfg']) if checkpoint else OutfitTransformerConfig(**cfg_kwargs)
        model = OutfitTransformer(cfg).cuda()

    elif model_type == 'clip':
        cfg = OutfitCLIPTransformerConfig(**state_dict['cfg']) if checkpoint else OutfitCLIPTransformerConfig(**cfg_kwargs)
        model = OutfitCLIPTransformer(cfg).cuda()
        
    if checkpoint:
        model.load_state_dict(
            state_dict['model'], strict=False
        )
        print(
            f"Loaded model from checkpoint: {checkpoint}"
        )
            
    return model