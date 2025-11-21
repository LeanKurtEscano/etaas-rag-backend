# backend/app/rag/ingest/product_ingestor.py
from typing import List, Dict
from schemas.product import Product

from backend.app.rag.embeddings.embedding import GeminiEmbedder
from backend.app.rag.vectorstore.vectore_store import PineconeVectorStore
from backend.app.utils.preprocess_product_json import preprocess_product


class ProductIngestor():

    def __init__(self, shop_id: str):
        self.shop_id = shop_id
        self.pinecone = PineconeVectorStore("products-index", embedder=GeminiEmbedder)

    def preprocess_to_store_embedding(self, product: Product) -> List[Dict]:
        preproccessed_chunks = preprocess_product(product, self.shop_id)
        self.pinecone.upsert_product_chunks(preproccessed_chunks)
    
       

   