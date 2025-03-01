import os
import random
import numpy as np
import torch
from typing import Iterable, Any, Optional
from itertools import islice
from tqdm import tqdm


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def batch_iterable(
    iterable: Iterable[Any],
    batch_size: int,
    desc: Optional[str] = None,
):
    iterator = iter(iterable)
    total = len(iterable) if hasattr(iterable, '__len__') else None
    
    pbar = tqdm(
        total=(total + batch_size - 1) // batch_size if total else None,
        desc=desc,
    )
    
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
        if pbar.total:  # Update progress only if total is known
            pbar.update(1)