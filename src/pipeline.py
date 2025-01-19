from fashion_recommenders.pipeline import BasePipeline
from fashion_recommenders import datatypes
from typing import List
from collections import defaultdict


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
        
        scores = self.model.predict(
            queries=[query]
        )[0]
        scores = float(scores)
        
        return scores


    def complementary_search(
        self,
        query: datatypes.FashionComplementaryQuery,
        k: int=12
    ) -> List[datatypes.FashionItem]:
        
        embeddings = self.model.embed_query(
            queries=[query]
        ) # List of (1, embedding_dim)
        embeddings = [e.detach().cpu().numpy() for e in embeddings] # List of (1, embedding_dim)
        
        # Reciprocal Search, Rank Fusion         
        results = self.indexer.multi_vector_search(
            embeddings=embeddings,
            k=k
        ) # List of num_query_items * k
        
        results = results[0]
        results = [self.loader.get_item(item_id) for item_id in results]
        
        return results