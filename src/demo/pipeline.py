
from abc import ABC, abstractmethod
from typing import List
from ..data import datatypes
        
        
class BasePipeline(ABC):
    
    @abstractmethod
    def compatibility_predict(
        self,
        queries: List[datatypes.FashionCompatibilityQuery],
    ) -> List[float]:
        
        raise NotImplementedError(
            'The cp method must be implemented by subclasses.'
        )

    @abstractmethod
    def complementary_search(
        self,
        queries: List[datatypes.FashionComplementaryQuery],
        k: int,
    ) -> List[List[datatypes.FashionItem]]:
        
        raise NotImplementedError(
            'The cir method must be implemented by subclasses.'
        )
        

class OutfitTransformerPipeline(BasePipeline):
    
    def __init__(
        self,
        model,
        loader,
        indexer
    ):
        self.model = model
        self.loader = loader
        self.indexer = indexer
    
    
    def compatibility_predict(
        self,
        query: datatypes.FashionCompatibilityQuery
    ) -> List[float]:
        
        scores = self.model.calculate_compatibility_score(
            queries=[query]
        )[0]
        scores = float(scores)
        
        return scores


    def complementary_search(
        self,
        query: datatypes.FashionComplementaryQuery,
        k: int=12
    ) -> List[datatypes.FashionItem]:
        
        embedding = self.model.embed_complementary_query(
            queries=[query]
        )[0].detach().cpu().numpy().tolist()
        
        # Reciprocal Search, Rank Fusion         
        results = self.indexer.search(
            embeddings=[embedding],
            k=k
        ) # List of num_query_items * k
        
        results = results[0]
        results = [self.loader.get_item(item_id) for item_id in results]
        
        return results