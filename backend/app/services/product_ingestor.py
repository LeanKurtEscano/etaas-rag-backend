from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.product import ProductRequest
from app.models.product import ProductMinimal
from app.rag.embeddings.embedding import GeminiEmbedder
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.utils.preprocess_product_json import preprocess_product


class ProductIngestor:

    def __init__(self, shop_id: str, db: AsyncSession):
        self.shop_id = shop_id
        self.db = db
        self.pinecone = PineconeVectorStore("products-index",dimension=3072, embedder=GeminiEmbedder())

    async def preprocess_to_store_embedding(self, product: ProductRequest) -> List[Dict]:
    
        db_product = ProductMinimal(name=product.name)
        self.db.add(db_product)
        await self.db.commit()        
        await self.db.refresh(db_product) 

       
        product.id = db_product.id


        preprocessed_chunks = preprocess_product(product, self.shop_id)
        print("Preprocessed chunks:", preprocessed_chunks)

       
        self.pinecone.upsert_product_chunks(preprocessed_chunks)

        return preprocessed_chunks
