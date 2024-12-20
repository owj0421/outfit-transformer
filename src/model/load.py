import torch
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitClipTransformerConfig,
    OutfitClipTransformer
)


def load_model(args):
    if args.model_type == 'original':
        cfg = OutfitTransformerConfig(
            nhead=16, #16
            num_layers=6, #6
            dim_feedforward=2048,
            dropout=0.3,
        )
        model = OutfitTransformer(
            cfg
        ).cuda()

    elif args.model_type == 'clip':
        cfg = OutfitClipTransformerConfig(
            nhead=16,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.3,
        )
        model = OutfitClipTransformer(
            cfg
        ).cuda()
        
    if args.checkpoint:
        model.load_state_dict(
            torch.load(
                args.checkpoint
            ),
            strict=False
        )
        print(
            f"Loaded model from checkpoint: {args.checkpoint}"
        )
            
    return model