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
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from typing import Literal
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from typing import Dict, Any, Optional


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def aggregate_embeddings(
    image_embeddings: Optional[Tensor] = None, 
    text_embeddings: Optional[Tensor] = None, 
    aggregation_method: str = 'concat'
) -> Tensor:
    """
    Aggregates image and text embeddings using the specified method.

    Args:
        image_embeds (Optional[Tensor]): Tensor containing image embeddings, shape (..., D).
        text_embeds (Optional[Tensor]): Tensor containing text embeddings, shape (..., D).
        aggregation_method (str): Method to aggregate embeddings ('concat' or 'mean').

    Returns:
        Tensor: Aggregated embeddings.
    """
    embeds = []
    if image_embeddings is not None:
        embeds.append(image_embeddings)
    if text_embeddings is not None:
        embeds.append(text_embeddings)

    if not embeds:
        raise ValueError('At least one of image_embeds or text_embeds must be provided.')

    if aggregation_method == 'concat':
        return torch.cat(embeds, dim=-1)
    elif aggregation_method == 'mean':
        return torch.mean(torch.stack(embeds), dim=-2)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}. Use 'concat' or 'mean'.")


def mean_pooling(
    model_output: Tensor, 
    attention_mask: Tensor
) -> Tensor:
    """
    Applies mean pooling on token embeddings, weighted by the attention mask.

    Args:
        model_output (Tensor): Output tensor from the transformer model, shape (batch_size, seq_length, hidden_size).
        attention_mask (Tensor): Attention mask tensor, shape (batch_size, seq_length).

    Returns:
        Tensor: Mean-pooled embeddings, shape (batch_size, hidden_size).
    """
    token_embeddings = model_output[0]  # First element of model_output contains the hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    mask_sum = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return summed_embeddings / mask_sum


class BaseImageEncoder(nn.Module, ABC):
    
    def __init__(
        self, 
        embedding_size: int = 64
    ):
        """
        Base class for embedding images into a fixed-size representation.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
        """
        super().__init__()
        self.embedding_size = embedding_size
        
    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model's parameters are stored.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(self.parameters()).device

    @abstractmethod
    def embed(
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
        return self.embed(images, *args, **kwargs)
    

class Resnet18ImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, 
        embedding_size: int = 64,
        size: int = 224,
        crop_size: int = 224
    ):
        """
        Image Encoder based on a pre-trained ResNet-18 model with a custom embedding layer.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
            image_size (int): Size to which each image is resized before center cropping.
            crop_size (int): Size of the center crop applied after resizing.
        """
        super().__init__(embedding_size=embedding_size)
        
        self.size = size
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load pre-trained ResNet-18 and adjust the final layer to match embedding_size
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=embedding_size
        )
    
    def embed(
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
        # Ensure all image sequences have the same length
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')

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
        embedding_size: int = 64,
        model_name_or_path: str = 'patrickjohncyh/fashion-clip',
        processor_args: Dict[str, Any] = None
    ):
        super().__init__(embedding_size=embedding_size)
        
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name_or_path
        )
        freeze_model(self.model)
        self.projection_dim = self.model.config.projection_dim
        self.processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path
        )
        self.processor_args = processor_args or {
            "return_tensors": "pt",
            # "input_data_format": "channels_first",
        }
    
    
    def embed(
       self, 
       images: List[List[Image.Image]]
    ):  
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')

        batch_size = len(images)
        images = sum(images, [])
        
        transformed_images = self.processor(
            images=images, 
            **self.processor_args
        ).to(self.device)
        image_embeddings = self.model(
            **transformed_images
        ).image_embeds
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.embedding_size
        )
        
        return image_embeddings
    
    
class BaseTextEncoder(nn.Module, ABC):
    def __init__(
        self, 
        embedding_size: int = 64
    ):
        """
        Base class for embedding text sequences into a fixed-size representation.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
        """
        super().__init__()
        self.embedding_size = embedding_size

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model's parameters are stored.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(self.parameters()).device

    @abstractmethod
    def embed(
        self, 
        texts: List[List[str]]
    ) -> torch.Tensor:
        """
        Abstract method for embedding a list of text sequences into a tensor.

        Args:
            texts (List[List[str]]): A batch of text sequences, each represented 
                as a list of strings.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, longest_sequence_length, embedding_size).
        """
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
        self, 
        texts: List[List[str]], 
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass that calls the embed method.

        Args:
            texts (List[List[str]]): A batch of text sequences, each represented 
                as a list of strings.
            *args, **kwargs: Additional arguments to be passed to the embed method.

        Returns:
            torch.Tensor: Output of the embed method.
        """
        return self.embed(texts, *args, **kwargs)
        
        
class HuggingFaceTextEncoder(BaseTextEncoder):
    
    def __init__(
        self,
        embedding_size: int = 64,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_args: Dict[str, Any] = None
    ):
        """
        Text Encoder using a Hugging Face transformer model, with a projection layer
        for dimensionality reduction.

        Args:
            embedding_size (int): Dimensionality of the output embedding.
            model_name_or_path (str): Pre-trained transformer model identifier or path.
            tokenizer_args (Dict[str, Any], optional): Arguments for the tokenizer.
                Defaults to a configuration with max length, padding, and truncation.
        """
        super().__init__(embedding_size=embedding_size)

        self.model = AutoModel.from_pretrained(model_name_or_path)
        freeze_model(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.proj = nn.Linear(
            in_features=self.model.config.hidden_size, 
            out_features=embedding_size
        )
        self.tokenizer_args = tokenizer_args or {
            'max_length': 16,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors': 'pt'
        }
        
    def embed(
        self, 
        texts: List[List[str]]
    ) -> Tensor:
        """
        Embeds a batch of text sequences into a tensor using a transformer model.

        Args:
            texts (List[List[str]]): Batch of text sequences, each represented as a list of strings.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, longest_sequence_length, embedding_size).
        """
        if len(set(len(text_seq) for text_seq in texts)) > 1:
            raise ValueError('All sequences in texts should have the same length.')

        batch_size = len(texts)
        texts = sum(texts, [])

        inputs = self.tokenizer(
            texts, **self.tokenizer_args
        )
        inputs = {
            key: value.to(self.device) 
            for key, value in inputs.items()
        }
        outputs = mean_pooling(
            model_output=self.model(**inputs), 
            attention_mask=inputs['attention_mask']
        )
        text_embeddings = self.proj(
            outputs
        )
        text_embeddings = text_embeddings.view(
            batch_size, -1, self.embedding_size
        )

        return text_embeddings
    
    
class CLIPTextEncoder(BaseTextEncoder):
    
    def __init__(
        self,
        embedding_size: int = 64,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_args: Dict[str, Any] = None
    ):
        super().__init__(embedding_size=embedding_size)
        
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path
        )
        freeze_model(self.model)
        self.projection_dim = self.model.config.projection_dim
        self.tokenizer = CLIPTokenizer.from_pretrained(
           model_name_or_path
        )
        self.tokenizer_args = tokenizer_args or {
            'max_length': 16,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors': 'pt'
        }
        
    def embed(
        self, 
        texts: List[List[str]]
    ) -> Tensor:
        if len(set(len(text_seq) for text_seq in texts)) > 1:
            raise ValueError('All sequences in texts should have the same length.')

        batch_size = len(texts)
        texts = sum(texts, [])
        
        inputs = self.tokenizer(
            texts, **self.tokenizer_args
        )
        inputs = {
            key: value.to(self.device) 
            for key, value in inputs.items()
        }
        # outputs = mean_pooling(
        #     model_output=self.model(**inputs), 
        #     attention_mask=inputs['attention_mask']
        # ) 
        # text_embeddings = self.proj(
        #     outputs
        # )
        text_embeddings = self.model(
            **inputs
        ).text_embeds
        text_embeddings = text_embeddings.view(
            batch_size, -1, self.embedding_size
        )    
            
        return text_embeddings