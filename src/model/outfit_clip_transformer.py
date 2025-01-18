from torch import nn
from dataclasses import dataclass

from typing import List, Tuple, Dict, Any, Union, Literal, Optional

from torch import Tensor
from PIL import Image
import numpy as np
import torch

import torch.nn.functional as F

import pickle
import os

import sys
from fashion_recommenders import datatypes
from fashion_recommenders.models.encoders.image import CLIPImageEncoder
from fashion_recommenders.models.encoders.text import CLIPTextEncoder
from fashion_recommenders.utils.model_utils import aggregate_embeddings

from . import outfit_transformer


PAD_IMAGE = Image.fromarray(
    np.zeros((224, 224, 3), dtype=np.uint8)
)
PAD_TEXT = ''
PAD_IMAGE_EMBEDDING = torch.zeros(512)
PAD_TEXT_EMBEDDING = torch.zeros(512)


# 현재 실행 중인 스크립트 파일의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class OutfitClipTransformerConfig:
    clip_huggingface_model_name: str = "patrickjohncyh/fashion-clip"# "Marqo/marqo-fashionSigLIP"
    query_image_path: str = os.path.join(current_dir, "../utils/question.jpg")
    
    embedding_size: int = 128
    d_encoder_output: int = 1024
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'
    
    nhead: int = 16
    dim_feedforward: int = 2024
    num_layers: int = 6
    dropout: float = 0.3
    normlaize_embeddings: bool = False
    
    def __post_init__(self):
        self.d_model = self.d_encoder_output
        
        if self.aggregation_method == 'concat':
            assert self.d_encoder_output % 2 == 0, (
                "If aggregation_method is 'concat', embedding_size must be even"
            )
            self.image_embedding_size = self.text_embedding_size = self.d_encoder_output // 2
        else:
            self.image_embedding_size = self.text_embedding_size = self.d_encoder_output


class OutfitClipTransformer(outfit_transformer.OutfitTransformer):
    
    def __init__(
        self,
        cfg: OutfitClipTransformerConfig = OutfitClipTransformerConfig()
    ):
        super().__init__()
        self.cfg = cfg
        self.query_image = Image.open(
            self.cfg.query_image_path
        )
        self.image_encoder = CLIPImageEncoder(
            embedding_size = cfg.image_embedding_size,
            model_name_or_path = cfg.clip_huggingface_model_name
        )
        self.text_encoder = CLIPTextEncoder(
            embedding_size = cfg.text_embedding_size,
            model_name_or_path = cfg.clip_huggingface_model_name
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