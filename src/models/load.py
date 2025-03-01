import torch
from typing import Any, Dict, Optional
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitCLIPTransformerConfig,
    OutfitCLIPTransformer
)
from torch.distributed import get_rank, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP


def load_model(model_type, checkpoint=None, **cfg_kwargs):
    is_distributed = torch.distributed.is_initialized()

    # 분산 학습 환경 설정
    if is_distributed:
        rank = get_rank()
        world_size = get_world_size()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    else:
        rank = 0
        world_size = 1
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 체크포인트 로드
    state_dict = None
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=map_location)
        cfg = state_dict.get('config', {})
        model_state_dict = state_dict.get('model', {})
    else:
        cfg = cfg_kwargs
        model_state_dict = None
    
    # 모델 초기화
    if model_type == 'original':
        model = OutfitTransformer(OutfitTransformerConfig(**cfg))
    elif model_type == 'clip':
        model = OutfitCLIPTransformer(OutfitCLIPTransformerConfig(**cfg))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.to(rank)
    
    # DDP 체크포인트와 일반 체크포인트 호환성 처리
    if model_state_dict:
        new_state_dict = {}
        for k, v in model_state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        if missing:
            print(f"[Warning] Missing keys in state_dict: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys in state_dict: {unexpected}")
        print(f"Loaded model from checkpoint: {checkpoint}")
    
    # DDP 적용 (가중치 로드 후 래핑)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], static_graph=True)
    
    return model