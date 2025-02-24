from typing import Optional
from torch import Tensor
import torch
from ..data.datatypes import FashionItem
from typing import List, Tuple
from torch import nn
import torch.nn.functional as F


def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def aggregate_embeddings(
    image_embeddings: Optional[Tensor] = None, 
    text_embeddings: Optional[Tensor] = None, 
    aggregation_method: str = 'concat'
) -> Tensor:
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
    token_embeddings = model_output[0]  # First element of model_output contains the hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    mask_sum = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return summed_embeddings / mask_sum