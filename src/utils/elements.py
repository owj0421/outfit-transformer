from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
from torch import Tensor

@dataclass
class Item:
    image: Image
    description: str
    category: Optional[str] = None
    id: Optional[str] = None
    
    
@dataclass
class Outfit:
    items: List[Item]
    
    def __call__(self):
        return self.items
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
    
    
@dataclass
class Query:
    query: str
    items: List[Item]
    
    def __call__(self):
        return self.items
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)