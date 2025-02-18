import torch
from torch import nn
from dataclasses import dataclass
from .modules.encoder import OutfitCLIPTransformerEncoder
from .outfit_transformer import OutfitTransformer, OutfitTransformerConfig


@dataclass
class OutfitCLIPTransformerConfig(OutfitTransformerConfig):
    enc_clip_model_name: str = "patrickjohncyh/fashion-clip"
            

class OutfitCLIPTransformer(OutfitTransformer):
    
    def __init__(
        self, 
        cfg: OutfitCLIPTransformerConfig = OutfitCLIPTransformerConfig()
    ):
        super().__init__(cfg)

    def _build_enc(self) -> OutfitCLIPTransformerEncoder:
        """Builds the outfit encoder using configuration parameters."""
        self.enc = OutfitCLIPTransformerEncoder(
            model_name=self.cfg.enc_clip_model_name,
            enc_norm_out=self.cfg.enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )