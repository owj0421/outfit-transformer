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


class BaseImageEncoder(nn.Module, ABC):
    
    def __init__(self):
        """
        Base class for embedding images into a fixed-size representation.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
        """
        super().__init__()
        
    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model's parameters are stored.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(self.parameters()).device

    @abstractmethod
    def _forward(
        self, 
        images: List[List[Image.Image]]
    ) -> torch.Tensor:
        """
        Abstract method for embedding a list of images into a tensor.

        Args:
            images (List[List[Image.Image]]): A batch of images, each represented 
                as a list of PIL Images.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, longest_outfit_length, embedding_size).
        """
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
        self, 
        images: List[List[Image.Image]], 
        normalize: bool = True,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass that calls the embed method.

        Args:
            images (List[List[Image.Image]]): A batch of images, each represented 
                as a list of PIL Images.
            *args, **kwargs: Additional arguments to be passed to the embed method.

        Returns:
            torch.Tensor: Output of the embed method.
        """
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')
        
        image_embeddings = self._forward(images, *args, **kwargs)
        
        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            
        return image_embeddings
    

class Resnet18ImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, 
        embedding_size: int = 64,
        size: int = 224,
        crop_size: int = 224,
        freeze: bool = False
    ):
        """
        Image Encoder based on a pre-trained ResNet-18 model with a custom embedding layer.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
            image_size (int): Size to which each image is resized before center cropping.
            crop_size (int): Size of the center crop applied after resizing.
        """
        super().__init__()

        # Load pre-trained ResNet-18 and adjust the final layer to match embedding_size
        self.embedding_size = embedding_size
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=embedding_size
        )
        if freeze:
            freeze_model(self.model)
            
        self.size = size
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _forward(
        self, 
        images: List[List[Image.Image]]
    ):  
        """
        Embeds a batch of images into a tensor using ResNet-18.

        Args:
            images (List[List[Image.Image]]): Batch of images, each represented as a list of PIL Images.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, longest_sequence_length, embedding_size).
        """
        batch_size = len(images)
        images = sum(images, [])
        
        transformed_images = torch.stack(
            [self.transform(image.convert('RGB')) for image in images]
        ).to(self.device)
        image_embeddings = self.model(
            transformed_images
        )
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.embedding_size
        )
        
        return image_embeddings
    
    
class CLIPImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, 
        model_name_or_path: str = 'patrickjohncyh/fashion-clip',
        freeze: bool = True
    ):
        super().__init__()
        # self.embedding_size = 512
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name_or_path
        )
        if freeze:
            freeze_model(self.model)
        self.embedding_size = self.model.config.projection_dim
        self.processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path
        )
    
    
    def _forward(
       self, 
       images: List[List[Image.Image]],
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
            batch_size, -1, self.embedding_size
        )

        return image_embeddings