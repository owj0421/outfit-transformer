from .datatypes import (
    FashionCompatibilityData,
    FashionFillInTheBlankData,
    FashionTripletData
)


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
    answers = [item['answers'] for item in batch]
    
    return FashionFillInTheBlankData(
        query=query,
        label=label,
        answers=answers
    )


def triplet_collate_fn(batch) -> FashionTripletData:
    query = [item['query'] for item in batch]
    answer = [item['answer'] for item in batch]
    
    return FashionTripletData(
        query=query,
        answer=answer
    )