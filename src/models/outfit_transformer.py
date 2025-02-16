from torch import nn
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from torch import Tensor
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os
import pathlib
from ..data import datatypes
from .modules.encoder import OutfitTransformerEncoder
from ..utils.model_utils import get_device

# Constants
PAD_IMAGE = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
PAD_TEXT = ''
PAD_IMAGE_EMBEDDING = torch.zeros(512)
PAD_TEXT_EMBEDDING = torch.zeros(512)

QUERY_IMG_PATH = pathlib.Path(__file__).parent.absolute() / 'question.jpg'


@dataclass
class OutfitTransformerConfig:
    padding: Literal['longest', 'max_length'] = 'longest'
    truncation: bool = True
    
    enc_text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    enc_dim_per_modality: int = 128
    enc_norm_out: bool = False
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'
    
    transformer_n_head: int = 16
    transformer_d_ffn: int = 2024
    transformer_n_layers: int = 6
    transformer_dropout: float = 0.3
    transformer_norm_out: bool = False
    
    d_embed: int = 128


class OutfitTransformer(nn.Module):
    query_img = Image.open(QUERY_IMG_PATH)
    
    def __init__(
        self, 
        cfg: OutfitTransformerConfig = OutfitTransformerConfig()
    ):
        super().__init__()
        self.cfg = cfg
        # Encoders
        self._build_enc()
        # Transformer Encoder
        self._build_transformer_enc()
        # Classifier and Embedding Layers
        self._build_calc_cp_ffn()
        self.classifier_embedding = nn.parameter.Parameter(
            torch.randn(1, self.enc.d_out, requires_grad=True)
        )
        self.embed_ffn = nn.Sequential(
            nn.Dropout(cfg.transformer_dropout),
            nn.Linear(self.enc.d_out, cfg.d_embed, bias=False)
        )
        
    def _build_enc(self) -> OutfitTransformerEncoder:
        """Builds the outfit encoder using configuration parameters."""
        self.enc = OutfitTransformerEncoder(
            text_model_name=self.cfg.enc_text_model_name,
            enc_dim_per_modality=self.cfg.enc_dim_per_modality,
            enc_norm_out=self.cfg.enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def _build_transformer_enc(self) -> nn.TransformerEncoder:
        """Builds the transformer encoder using configuration parameters."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.enc.d_out,
            nhead=self.cfg.transformer_n_head,
            dim_feedforward=self.cfg.transformer_d_ffn,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.transformer_enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.cfg.transformer_n_layers)
    
    def _build_calc_cp_ffn(self) -> nn.Sequential:
        """Builds the feed-forward classifier layer."""
        self.calc_cp_ffn = nn.Sequential(
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.enc.d_out, 1, bias=False),
            nn.Sigmoid()
        )
    
    def _pad_and_mask(self, outfits) -> Tuple[List, List, Tensor]:
        """Pads images, texts, and masks to the specified length."""
        images, texts, mask = [], [], []
        
        max_length = (
            max([len(outfit) for outfit in outfits]) if self.cfg.padding == 'longest'
            else min(max_length, max([len(outfit) for outfit in outfits]))
        )
        for outfit in outfits:
            if self.cfg.truncation:
                outfit = outfit[:max_length]
            images.append(
                [item.image for item in outfit] + [PAD_IMAGE] * (max_length - len(outfit))
            )
            texts.append(
                [f"Category: {item.category}; Description: {item.description}" for item in outfit] + [PAD_TEXT] * (max_length - len(outfit))
            )
            mask.append(
                [0] * len(outfit) + [1] * (max_length - len(outfit))
            )
            
        return images, texts, torch.BoolTensor(mask).to(self.device)
    
    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return get_device(self)
    
    def calculate_compatibility_score(
        self, 
        queries: List[datatypes.FashionCompatibilityQuery], 
        *args, **kwargs
    ) -> Tensor:
        """
        Predicts the compatibility scores for the given queries.
        """
        bsz = len(queries)
        outfits = [query.outfit for query in queries]
        
        images, texts, mask = self._pad_and_mask(outfits)
        enc_outs = self.enc(images, texts)
        
        enc_outs = torch.cat(
            [self.classifier_embedding.unsqueeze(0).expand(bsz, -1, -1), enc_outs], dim=1
        )
        mask = torch.cat(
            [torch.zeros(bsz, 1, dtype=torch.bool, device=self.device), mask], dim=1
        )
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        scores = self.calc_cp_ffn(last_hidden_states[:, 0, :])

        return scores
    
    def embed_complementary_query(
        self, 
        query: List[datatypes.FashionComplementaryQuery], 
        *args, **kwargs
    ) -> List[Tensor]:
        """
        Embeds query items for compatibility.
        """
        query_items = [datatypes.FashionItem(category=i.category, image=self.query_img, description=i.category) for i in query]
        outfits = [[query_item] + i.outfit for query_item, i in zip(query_items, query)]
        
        images, texts, mask = self._pad_and_mask(outfits)
        enc_outs = self.enc(images, texts)
        
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return [embedding for embedding in embeddings]

    def embed_complementary_item(
        self, 
        item: List[datatypes.FashionItem], 
        *args, **kwargs
    ) -> List[Tensor]:
        """
        Embeds candidate items for compatibility.
        """
        outfits = [[item_] for item_ in item]
        
        images, texts, mask = self._pad_and_mask(outfits)
        enc_outs = self.enc(images, texts)
        
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return [embedding for embedding in embeddings]