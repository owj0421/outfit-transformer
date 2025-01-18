from fashion_recommenders.pipeline import BasePipeline
from fashion_recommenders import datatypes
from typing import List


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
        queries: List[datatypes.FashionCompatibilityQuery]
    ) -> List[float]:
        
        scores = self.model.predict(
            queries=queries
        )
        scores = list(map(float, scores))
        
        return scores


    def complementary_search(
        self,
        queries: List[datatypes.FashionComplementaryQuery],
        k: int=12
    ) -> List[List[datatypes.FashionItem]]:
        
        embeddings = self.model.embed_query(
            queries=queries
        ).detach().cpu().numpy() # (n_queries, emb_dim)
        
        results = self.indexer.search(
            embeddings=embeddings,
            k=k
        ) # (n_queries, k)
        
        results = [
            [self.loader.get_item(item_id) for item_id in result]
            for result in results 
        ]
        
        return results