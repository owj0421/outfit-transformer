# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    CLIPTokenizer, 
    CLIPTextModelWithProjection,
)
from typing import Literal
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from typing import Dict, Any, Optional

from ...utils.model_utils import freeze_model, mean_pooling
    
    
class BaseTextEncoder(nn.Module, ABC):
    def __init__(self):
        """
        Base class for embedding text sequences into a fixed-size representation.

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
    def encode(
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
        return self.encode(texts, *args, **kwargs)
        
        
class HuggingFaceTextEncoder(BaseTextEncoder):
    
    def __init__(
        self,
        embedding_size: int = 64,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = True
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
        super().__init__()
        self.embedding_size=embedding_size
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if freeze:
            freeze_model(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.proj = nn.Linear(
            in_features=self.model.config.hidden_size, 
            out_features=embedding_size
        )
        
    def encode(
        self, 
        texts: List[List[str]],
        tokenizer_kargs: Dict[str, Any] = None
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

        tokenizer_kargs = tokenizer_kargs if tokenizer_kargs is not None else {
            'max_length': 16,
            'padding': 'max_length',
            'truncation': True,
        }
        
        tokenizer_kargs['return_tensors'] = 'pt'
        
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
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = True
    ):
        super().__init__()
        self.embedding_size = 512
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path
        )
        if freeze:
            freeze_model(self.model)
        self.projection_dim = self.model.config.projection_dim
        self.tokenizer = CLIPTokenizer.from_pretrained(
           model_name_or_path
        )
        
    def encode(
        self, 
        texts: List[List[str]],
        tokenizer_kargs: Dict[str, Any] = None
    ) -> Tensor:
        if len(set(len(text_seq) for text_seq in texts)) > 1:
            raise ValueError('All sequences in texts should have the same length.')

        batch_size = len(texts)
        texts: List[str] = sum(texts, []) # 
        
        tokenizer_kargs = tokenizer_kargs if tokenizer_kargs is not None else {
            'max_length': 16,
            'padding': 'max_length',
            'truncation': True,
        }
        tokenizer_kargs['return_tensors'] = 'pt'
        
        inputs = self.tokenizer(
            text=texts, **tokenizer_kargs
        )
        
        inputs = {
            key: value.to(self.device) 
            for key, value in inputs.items()
        }
        
        text_embeddings = self.model(
            **inputs
        ).text_embeds
        
        text_embeddings = text_embeddings.view(
            batch_size, -1, self.embedding_size
        )    
            
        return text_embeddings