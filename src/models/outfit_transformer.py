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
from ..data.datatypes import (
    FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem
)
from .modules.encoder import OutfitTransformerEncoder
from ..utils.model_utils import get_device

# Constants
# Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
PAD_IMAGE = np.array(Image.new("RGB", (224, 224)))
PAD_TEXT = ''

QUERY_IMG_PATH = pathlib.Path(__file__).parent.absolute() / 'question.jpg'


@dataclass
class OutfitTransformerConfig:
    padding: Literal['longest', 'max_length'] = 'longest'
    max_length: int = 16
    truncation: bool = True
    
    init_enc: bool = True
    enc_text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    enc_dim_per_modality: int = 128
    enc_norm_out: bool = True
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'
    
    init_transformer: bool = True
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
        # Outfit Encoder
        if self.cfg.init_enc:
            print("Building Backbone Encoder...")
            self._build_enc()
        # Transformer Encoder
        if self.cfg.init_transformer:
            print("Building Transformer Encoder...")
            self._build_transformer_enc()
            # Classifier and Embedding Layers
            self._build_calc_cp_ffn()
            self._build_embed_ffn()
            self.classifier_embedding = nn.parameter.Parameter(
                torch.randn(1, self.enc.d_out, requires_grad=True)
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
        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.enc.d_out,
            nhead=self.cfg.transformer_n_head,
            dim_feedforward=self.cfg.transformer_d_ffn,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer, 
            num_layers=self.cfg.transformer_n_layers
        )
    
    def _build_calc_cp_ffn(self) -> nn.Sequential:
        """Builds the feed-forward classifier layer."""
        self.calc_cp_ffn = nn.Sequential(
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.enc.d_out, 1, bias=False),
            nn.Sigmoid()
        )
        
    def _build_embed_ffn(self) -> nn.Sequential:
        """Builds the feed-forward embedding layer."""
        self.embed_ffn = nn.Sequential(
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.enc.d_out, self.cfg.d_embed, bias=False)
        )
    
    def _pad_and_mask(self, outfits) -> Tuple[List, List, Tensor]:
        if self.cfg.padding == 'max_length':
            max_length = self.cfg.max_length
        else:
            max_length = max(len(o) for o in outfits)
            if self.cfg.truncation:
                max_length = min(self.cfg.max_length, max_length)
        
        images, texts, mask = [], [], []
        for outfit in outfits:
            outfit = outfit[:max_length]
            images.append(
                [item.image for item in outfit] + [PAD_IMAGE] * (max_length - len(outfit))
            )
            texts.append(
                [f"A {item.category} featuring {item.description}" for item in outfit] + [PAD_TEXT] * (max_length - len(outfit))
            )
            mask.append(
                [0] * len(outfit) + [1] * (max_length - len(outfit))
            )
            
        return images, texts, torch.BoolTensor(mask).to(self.device)
    
    def _pad_and_mask_for_embedding(self, es_of_outfits):
        if self.cfg.padding == 'max_length':
            max_length = self.cfg.max_length
        else:
            max_length = max(len(es) for es in es_of_outfits)
            if self.cfg.truncation:
                max_length = min(self.cfg.max_length, max_length)

        batch_size = len(es_of_outfits)

        # Initialize tensors with correct dtype and device
        embeddings = torch.zeros((batch_size, max_length, self.enc.d_out), dtype=torch.float, device=self.device)
        mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=self.device)

        for i, es_of_outfit in enumerate(es_of_outfits):
            es_of_outfit = np.array(es_of_outfit[:max_length])
            length = len(es_of_outfit)
            embeddings[i, :length] = torch.tensor(es_of_outfit, dtype=torch.float, device=self.device)
            mask[i, :length] = False

        return embeddings, mask
    
    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return get_device(self)
    
    def forward(
        self, 
        inputs: Union[
            List[FashionCompatibilityQuery], 
            List[FashionComplementaryQuery],
            List[FashionItem],
        ], 
        *args, **kwargs
    ) -> Tensor:
        if isinstance(inputs[0], FashionCompatibilityQuery):
            return self.calculate_compatibility_score(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionComplementaryQuery):
            return self.embed_complementary_query(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionItem):
            return self.embed_complementary_item(inputs, *args, **kwargs)
        else:
            raise ValueError("Invalid input type.")
    
    def calculate_compatibility_score(
        self, 
        query: List[FashionCompatibilityQuery],
        use_precomputed_embedding: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """
        Predicts the compatibility scores for the given queries.
        """
        outfits = [query_.outfit for query_ in query]
        
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            es_of_outfits = [[item_.embedding for item_ in outfit] for outfit in outfits]
            enc_outs, mask = self._pad_and_mask_for_embedding(es_of_outfits)
        else:
            outfits = [query_.outfit for query_ in query]
            images, texts, mask = self._pad_and_mask(outfits)
            enc_outs = self.enc(images, texts)

        enc_outs = torch.cat(
            [self.classifier_embedding.unsqueeze(0).expand(len(query), -1, -1), enc_outs], dim=1
        )
        mask = torch.cat(
            [torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), mask], dim=1
        )
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        scores = self.calc_cp_ffn(last_hidden_states[:, 0, :])

        return scores
    
    def embed_complementary_query(
        self, 
        query: List[FashionComplementaryQuery], 
        use_precomputed_embedding: bool=False,
        *args, **kwargs
    ) -> List[Tensor]:
        """
        Embeds query items for compatibility.
        """
        query_items = [FashionItem(category=i.category, image=self.query_img, description=i.category) for i in query]
        outfits = [query_.outfit for query_ in query]
        
        if use_precomputed_embedding:
            query_images, query_texts, query_mask = self._pad_and_mask(query_items)
            query_enc_outs = self.enc(query_images, query_texts)
            
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            es_of_outfits = [[item_.embedding for item_ in outfit] for outfit in outfits]
            enc_outs, mask = self._pad_and_mask_for_embedding(es_of_outfits)
            
            enc_outs = torch.cat([query_enc_outs, enc_outs], dim=1)
            mask = torch.cat([query_mask, mask], dim=1)
        else:
            outfits = [[query_item] + outfit for query_item, outfit in zip(query_items, outfits)]
            images, texts, mask = self._pad_and_mask(outfits)
            enc_outs = self.enc(images, texts)
        
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return [embedding for embedding in embeddings]

    def embed_complementary_item(
        self, 
        item: List[FashionItem], 
        use_precomputed_embedding: bool=False,
        *args, **kwargs
    ) -> List[Tensor]:
        """
        Embeds candidate items for compatibility.
        """
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in item])
            es_of_outfits = [[item_.embedding] for item_ in item]
            enc_outs, mask = self._pad_and_mask_for_embedding(es_of_outfits)
        else:
            assert item
            outfits = [[item_] for item_ in item]
            images, texts, mask = self._pad_and_mask(outfits)
            enc_outs = self.enc(images, texts)
        
        last_hidden_states = self.transformer_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return [embedding[0] for embedding in embeddings]