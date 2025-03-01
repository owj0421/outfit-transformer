from typing import List

from .datatypes import (
    FashionCompatibilityData,
    FashionFillInTheBlankData,
    FashionTripletData,
    FashionItem
)


def item_collate_fn(batch) -> List[FashionItem]:
    return [item for item in batch]


def cp_collate_fn(batch) -> FashionCompatibilityData:
    label = [item['label'] for item in batch]
    query = [item['query'] for item in batch]
    
    return FashionCompatibilityData(
        label=label,
        query=query
    )
    

def fitb_collate_fn(batch) -> FashionFillInTheBlankData:
    query = [item['query'] for item in batch]
    label = [item['label'] for item in batch]
    candidates = [item['candidates'] for item in batch]
    
    return FashionFillInTheBlankData(
        query=query,
        label=label,
        candidates=candidates
    )


def triplet_collate_fn(batch) -> FashionTripletData:
    query = [item['query'] for item in batch]
    answer = [item['answer'] for item in batch]
    
    return FashionTripletData(
        query=query,
        answer=answer
    )