from torch import nn
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from torch import Tensor
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import pathlib
from ..data.datatypes import (
    FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem
)
from .modules.encoder import ItemEncoder
from ..utils.model_utils import get_device

@dataclass
class OutfitTransformerConfig:
    padding: Literal['longest', 'max_length'] = 'longest'
    max_length: int = 16
    truncation: bool = True
    
    item_enc_text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    item_enc_dim_per_modality: int = 128
    item_enc_norm_out: bool = True
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'
    
    transformer_n_head: int = 16 # Original: 16
    transformer_d_ffn: int = 2024 # Original: Unknown
    transformer_n_layers: int = 6 # Original: 6
    transformer_dropout: float = 0.3 # Original: Unknown
    transformer_norm_out: bool = False
    
    d_embed: int = 128


class OutfitTransformer(nn.Module):
    
    def __init__(self, cfg: Optional[OutfitTransformerConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else OutfitTransformerConfig()
        self._init_item_enc()
        self._init_style_enc()
        self._init_variables()
        
    def _init_item_enc(self):
        """Builds the outfit encoder using configuration parameters."""
        self.item_enc = ItemEncoder(
            text_model_name=self.cfg.item_enc_text_model_name,
            enc_dim_per_modality=self.cfg.item_enc_dim_per_modality,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def _init_style_enc(self):
        """Builds the transformer encoder using configuration parameters."""
        style_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.item_enc.d_embed,
            nhead=self.cfg.transformer_n_head,
            dim_feedforward=self.cfg.transformer_d_ffn,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.style_enc = nn.TransformerEncoder(
            encoder_layer=style_enc_layer,
            num_layers=self.cfg.transformer_n_layers,
            enable_nested_tensor=False
        )
        self.predict_ffn = nn.Sequential(
            # nn.LayerNorm(self.item_enc.d_embed),
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.item_enc.d_embed, 1),
            nn.Sigmoid()
        )
        self.embed_ffn = nn.Sequential(
            nn.Linear(self.item_enc.d_embed, self.cfg.d_embed, bias=False)
        )
    
    def _init_variables(self):
        image_size = (self.item_enc.image_size, self.item_enc.image_size)
        # self.image_query = Image.open(self.cfg.query_img_path).resize(image_size)
        self.image_pad = Image.new("RGB", image_size)
        self.text_pad = ''
        
        self.task_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        self.predict_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        self.embed_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        self.pad_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed) * 0.02, requires_grad=True
        )
    
    def _get_max_length(self, sequences):
        if self.cfg.padding == 'max_length':
            return self.cfg.max_length
        max_length = max(len(seq) for seq in sequences)
        
        return min(self.cfg.max_length, max_length) if self.cfg.truncation else max_length

    def _pad_sequences(self, sequences, pad_value, max_length):
        return [seq[:max_length] + [pad_value] * (max_length - len(seq)) for seq in sequences]

    def _pad_and_mask_for_outfits(self, outfits):
        max_length = self._get_max_length(outfits)
        images = self._pad_sequences(
            [[item.image for item in outfit] for outfit in outfits], 
            self.image_pad, max_length
        )
        texts = self._pad_sequences(
            [[f"{item.description}" for item in outfit] for outfit in outfits], 
            self.text_pad, max_length
        )
        mask = [[0] * len(seq) + [1] * (max_length - len(seq)) for seq in outfits]
        
        return images, texts, torch.BoolTensor(mask).to(self.device)
    
    def _pad_and_mask_for_embs(self, embs_of_outfits):
        max_length = self._get_max_length(embs_of_outfits)
        batch_size = len(embs_of_outfits)

        embeddings = torch.empty((batch_size, max_length, self.item_enc.d_embed), 
                                 dtype=torch.float, device=self.device)
        mask = []

        for i, embs_of_outfit in enumerate(embs_of_outfits):
            embs_of_outfit = torch.tensor(
                np.array(embs_of_outfit[:max_length]), dtype=torch.float
            ).to(self.device)
            length = len(embs_of_outfit)

            embeddings[i, :length] = embs_of_outfit
            embeddings[i, length:] = self.pad_emb  # 패딩 부분을 학습 가능한 벡터로 채움
            mask.append([0] * length + [1] * (max_length - length))
        
        return embeddings, torch.BoolTensor(mask).to(self.device)
    
    def _style_enc_forward(self, embs_of_inputs, src_key_padding_mask):
        if self.cfg.aggregation_method == 'concat':
            half_d_embed = self.item_enc.d_embed // 2
            normalized_embs = torch.cat([
                F.normalize(embs_of_inputs[:, :, :half_d_embed], p=2, dim=-1),
                F.normalize(embs_of_inputs[:, :, half_d_embed:], p=2, dim=-1)
            ], dim=-1)
        else:
            normalized_embs = F.normalize(embs_of_inputs, p=2, dim=-1)
        
        return self.style_enc(normalized_embs, src_key_padding_mask=src_key_padding_mask)
    
    def predict_score(self, query: List[FashionCompatibilityQuery], use_precomputed_embedding: bool = False) -> Tensor:
        outfits = [query_.outfit for query_ in query]
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            embs_of_inputs = [[item_.embedding for item_ in outfit] for outfit in outfits]
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            outfits = [query_.outfit for query_ in query]
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
            
        task_emb = torch.cat([self.task_emb, self.predict_emb], dim=-1)
        
        embs_of_inputs = torch.cat([
            task_emb.view(1, 1, -1).expand(len(query), -1, -1), # [B, 1, D]
            embs_of_inputs # [B, L, D]
        ], dim=1) # [B, L+1, D]
        mask = torch.cat([
            torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), # [B, 1]
            mask # [B, L]
        ], dim=1) # [B, L+1]
        
        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        scores = self.predict_ffn(last_hidden_states[:, 0, :])
        
        return scores
    
    def embed_query(self, query: List[FashionComplementaryQuery], use_precomputed_embedding: bool=False) -> Tensor:
        # q_items = [[FashionItem(category=i.category, image=self.image_query, description=i.category)] for i in query]
        outfits = [query_.outfit for query_ in query]
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            embs_of_inputs = [[item_.embedding for item_ in outfit] for outfit in outfits]
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
            
        task_emb = torch.cat([self.task_emb, self.embed_emb], dim=-1)
        
        embs_of_inputs = torch.cat([
            task_emb.view(1, 1, -1).expand(len(query), -1, -1), embs_of_inputs
        ], dim=1)
        mask = torch.cat([
            torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), mask
        ], dim=1)

        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        return F.normalize(embeddings, p=2, dim=-1) if self.cfg.transformer_norm_out else embeddings

    def embed_item(self, item: List[FashionItem], use_precomputed_embedding: bool=False) -> Tensor:
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in item])
            embs_of_inputs = [[item_.embedding] for item_ in item]
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            outfits = [[item_] for item_ in item]
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
        
        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :]) # [B, D]
            
        return F.normalize(embeddings, p=2, dim=-1) if self.cfg.transformer_norm_out else embeddings

    def forward(
        self, 
        inputs: List[Union[FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem]],
        *args, **kwargs
    ) -> Tensor:
        if isinstance(inputs[0], FashionCompatibilityQuery):
            return self.predict_score(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionComplementaryQuery):
            return self.embed_query(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionItem):
            return self.embed_item(inputs, *args, **kwargs)
        else:
            raise ValueError("Invalid input type.")
        
    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return get_device(self)