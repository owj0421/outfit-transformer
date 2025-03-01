# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection, 
)
from typing import Literal
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from typing import Dict, Any, Optional

from ...utils.model_utils import freeze_model, mean_pooling

import numpy as np

class BaseImageEncoder(nn.Module, ABC):
    
    def __init__(self):
        super().__init__()
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    @abstractmethod
    def image_size(self) -> int:
        raise NotImplementedError('The image_size property must be implemented by subclasses.')
    
    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError('The d_embed property must be implemented by subclasses.')

    @abstractmethod
    def _forward(
        self, 
        images: List[List[np.ndarray]]
    ) -> torch.Tensor:
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
        self, 
        images: List[List[np.ndarray]], 
        normalize: bool = True,
        *args, **kwargs
    ) -> torch.Tensor:
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')
        
        image_embeddings = self._forward(images, *args, **kwargs)
        
        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            
        return image_embeddings
    

class Resnet18ImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, d_embed: int = 64,
        size: int = 224, crop_size: int = 224, freeze: bool = False
    ):
        super().__init__()

        # Load pre-trained ResNet-18 and adjust the final layer to match d_embed
        self.d_embed = d_embed
        self.size = size
        self.crop_size = crop_size
        self.freeze = freeze
        
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=d_embed
        )
        if freeze:
            freeze_model(self.model)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @property
    def image_size(self) -> int:
        return self.crop_size
    
    @property
    def d_embed(self) -> int:
        return self.d_embed
    
    def _forward(
        self, 
        images: List[List[np.ndarray]]
    ):  
        batch_size = len(images)
        images = sum(images, [])
        
        transformed_images = torch.stack(
            [self.transform(image) for image in images]
        ).to(self.device)
        image_embeddings = self.model(
            transformed_images
        )
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.d_embed
        )
        
        return image_embeddings
    
    
class CLIPImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, 
        model_name_or_path: str = 'patrickjohncyh/fashion-clip',
        freeze: bool = True
    ):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name_or_path, weights_only=False
        )
        self.model.eval()
        if freeze:
            freeze_model(self.model)
        self.processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path, do_convert_rgb=False
        )
        
    @property
    def image_size(self) -> int:
        return self.processor.size['shortest_edge']
    
    @property
    def d_embed(self) -> int:
        return self.model.config.projection_dim
    
    @torch.no_grad()
    def _forward(
       self, 
       images: List[List[np.ndarray]],
       processor_kargs: Dict[str, Any] = None
    ):  
        batch_size = len(images)
        images = sum(images, [])
        
        processor_kargs = processor_kargs if processor_kargs is not None else {}
        processor_kargs['return_tensors'] = 'pt'
        
        transformed_images = self.processor(
            images=images, **processor_kargs
        ).to(self.device)
        
        image_embeddings = self.model(
            **transformed_images
        ).image_embeds
        
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.d_embed
        )

        return image_embeddings