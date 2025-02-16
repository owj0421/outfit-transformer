from typing import Optional
from torch import Tensor
import torch


def get_device(model: torch.nn.Module) -> torch.device:
    """Gets the device on which the model's parameters are stored."""
    return next(model.parameters()).device


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