from fashion_recommenders.models.pipeline import BaseCPPipeline, BaseCIRPipeline
from fashion_recommenders.utils.elements import Outfit, Item, Query
from typing import List


class OutfitTransformerCPPipeline(BaseCPPipeline):

    def predict(
        self,
        outfits: List[Outfit]
    ) -> List[float]:
        
        scores = self.model.predict(
            outfits=outfits
        )
        scores = list(map(float, scores))
        
        return scores
    

class OutfitTransformerCIRPipeline(BaseCIRPipeline):

    def search(
        self,
        queries: List[Query],
        k: int=12
    ) -> List[List[Item]]:
        
        embeddings = self.model.embed_query(
            query=queries
        ).detach().cpu().numpy() # (n_queries, emb_dim)
        
        results = self.indexer.search(
            embeddings=embeddings,
            k=k
        ) # (n_queries, k)
        
        results = [
            [self.loader(str(item_id)) for item_id in result]
            for result in results 
        ]
        
        return results