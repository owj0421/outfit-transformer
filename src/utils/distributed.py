import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# 윈도우 플랫폼에서 torch.distributed 패키지는
# Gloo backend, FileStore 및 TcpStore 만을 지원합니다.
# FileStore의 경우, init_process_group 에서
# init_method 매개변수를 로컬 파일로 설정합니다.
# 다음 예시:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# TcpStore의 경우 리눅스와 동일한 방식입니다.

def setup(
    rank: int, world_size: int
):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()