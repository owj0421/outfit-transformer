from torch import nn
from dataclasses import dataclass

from typing import List, Tuple, Dict, Any, Union, Literal, Optional

from torch import Tensor
from PIL import Image
import numpy as np
import torch

import torch.nn.functional as F

import os
import sys
from fashion_recommenders import datatypes
from fashion_recommenders.models.encoders.image import Resnet18ImageEncoder
from fashion_recommenders.models.encoders.text import HuggingFaceTextEncoder
from fashion_recommenders.utils.model_utils import aggregate_embeddings


PAD_IMAGE = Image.fromarray(
    np.zeros((224, 224, 3), dtype=np.uint8)
)
PAD_TEXT = ''
PAD_IMAGE_EMBEDDING = torch.zeros(512)
PAD_TEXT_EMBEDDING = torch.zeros(512)


# 현재 실행 중인 스크립트 파일의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class OutfitTransformerConfig:
    query_image_path: str = os.path.join(current_dir, "../utils/question.jpg")
    
    embedding_size: int = 128
    d_encoder_output: int = 128
    huggingface_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'
    
    nhead: int = 16
    dim_feedforward: int = 2024
    num_layers: int = 6
    dropout: float = 0.3
    normlaize_embeddings: bool = False
    
    def __post_init__(self):
        self.d_model = self.d_encoder_output
        
        if self.aggregation_method == 'concat':
            assert self.embedding_size % 2 == 0, (
                "If aggregation_method is 'concat', embedding_size must be even"
            )
            self.image_embedding_size = self.text_embedding_size = self.d_encoder_output // 2
        else:
            self.image_embedding_size = self.text_embedding_size = self.d_encoder_output


class OutfitTransformer(nn.Module):
    
    def __init__(
        self,
        cfg: OutfitTransformerConfig = OutfitTransformerConfig()
    ):
        super().__init__()
        self.cfg = cfg
        self.query_image = Image.open(
            self.cfg.query_image_path
        )
        self.image_encoder = Resnet18ImageEncoder(
            embedding_size = cfg.image_embedding_size
        )
        self.text_encoder = HuggingFaceTextEncoder(
            embedding_size = cfg.text_embedding_size,
            model_name_or_path = cfg.huggingface_model_name
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.transformer_encoder=nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.classifier_ffn = nn.Sequential(
            nn.Dropout(self.cfg.dropout),
            nn.Linear(cfg.d_model, 1, bias=False),
            nn.Sigmoid()
        )
        self.classifier_embedding = nn.parameter.Parameter(
            torch.randn(1, cfg.d_encoder_output, requires_grad=True)
        )
        self.embed_mlp = nn.Sequential(
            nn.Dropout(self.cfg.dropout),
            nn.Linear(cfg.d_model, cfg.embedding_size, bias=False)
        )
        
    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model's parameters are stored.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(self.parameters()).device
        
    def encode(
        self,
        outfits: List[List[datatypes.FashionItem]],
        padding: Literal['longest', 'max_length'] = 'longest',
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        assert not ((padding == 'max_length') and (max_length is None)), (
            "If padding is 'max_length', max_length must be provided"
        )
        assert not ((padding == 'longest') and (truncation is True)), (
            "If padding is 'longest', truncation must be False"
        )
        
        max_length = (
            max([len(outfit) for outfit in outfits])
            if padding == 'longest' else min(max_length, max([len(outfit) for outfit in outfits]))
        )

        images, texts, mask = [], [], []
        for outfit in outfits:
            if truncation:
                outfit = outfit[:max_length]
                
            images.append(
                [item.image for item in outfit] \
                + [PAD_IMAGE for _ in range(max_length - len(outfit))]
            )
            texts.append(
                [item.description for item in outfit] \
                + [PAD_TEXT for _ in range(max_length - len(outfit))]
            )
            mask.append(
                [0 for _ in outfit] \
                + [1 for _ in range(max_length - len(outfit))]
            )

        image_encoder_outputs = self.image_encoder(images)
        text_encoder_outputs = self.text_encoder(texts)
            
        encoder_outputs = aggregate_embeddings(
            image_embeddings=image_encoder_outputs,
            text_embeddings=text_encoder_outputs,
            aggregation_method='concat'
        )
        
        encoder_outputs = F.normalize(encoder_outputs, p=2, dim=-1)
        
        mask = torch.BoolTensor(mask).to(self.device)
        
        return encoder_outputs, mask
    
    def predict(
        self,
        queries: List[datatypes.FashionCompatibilityQuery],
        *args, **kwargs
    ) -> Tensor:
        
        assert isinstance(queries, list), (
            "outfits must be a list of Outfit instances"
        )
        
        bsz = len(queries)
        
        outfits = [query.outfit for query in queries]
        encoder_outputs, mask = self.encode(
            outfits, *args, **kwargs
        )
        encoder_outputs = torch.cat(
            [
                self.classifier_embedding.unsqueeze(0).expand(bsz, -1, -1), 
                encoder_outputs
            ], dim=1
        )
        mask = torch.cat(
            [
                torch.zeros(bsz, 1, dtype=torch.bool, device=self.device), 
                mask
            ], dim=1
        )
        last_hidden_states = self.transformer_encoder(
            encoder_outputs, 
            src_key_padding_mask=mask
        )
        outputs = self.classifier_ffn(
            last_hidden_states[:, 0, :]
        )
        
        return outputs
    
    def embed_query(
        self,
        queries: List[datatypes.FashionComplementaryQuery],
        *args, **kwargs
    ) -> Tensor:
        query_items = [
            datatypes.FashionItem(category=i.category, image=self.query_image, description=i.category)
            for i in queries
        ]
        outfits = [
            [query_item] + i.outfit
            for query_item, i in zip(query_items, queries)
        ]
        encoder_outputs, mask = self.encode(
            outfits, *args, **kwargs
        )
        last_hidden_states = self.transformer_encoder(
            encoder_outputs, 
            src_key_padding_mask=mask
        )
        embeddings = self.embed_mlp(
            last_hidden_states[:, 0, :]
        )
        if self.cfg.normlaize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Match the shape
        # (bsz, embedding_dim) to list of (1, embedding_dim)
        embeddings = [embedding.unsqueeze(0) for embedding in embeddings]
        
        return embeddings
    
    def embed_item(
        self,
        items: List[datatypes.FashionItem],
        *args, **kwargs
    ) -> Tensor:
        outfits = [[item] for item in items]
        encoder_outputs, mask = self.encode(
            outfits, *args, **kwargs
        )
        last_hidden_states = self.transformer_encoder(
            encoder_outputs, 
            src_key_padding_mask=mask
        )
        embeddings = self.embed_mlp(
            last_hidden_states[:, 0, :]
        )
        if self.cfg.normlaize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings