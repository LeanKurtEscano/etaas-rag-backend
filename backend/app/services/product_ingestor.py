from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.product import Product, ProductRequest
from app.models.product import ProductMinimal
from app.rag.embeddings.embedding import GeminiEmbedder
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.utils.preprocess_product_json import preprocess_product


class ProductIngestor:

    def __init__(self, shop_id: str, db: AsyncSession):
        self.shop_id = shop_id
        self.db = db
        self.pinecone = PineconeVectorStore(
            "products-index",
            dimension=3072,
            embedder=GeminiEmbedder()
        )

    async def preprocess_to_store_embedding(self, product: ProductRequest) -> List[Dict]:

        # Insert minimal DB product row
        db_product = ProductMinimal(name=product.name, uid = product.uid)
        self.db.add(db_product)
        await self.db.commit()
        await self.db.refresh(db_product)

      
        product.id = db_product.id

       
        preprocessed_chunks = preprocess_product(product, self.shop_id)
        print("Preprocessed chunks:", preprocessed_chunks)

       
        self.pinecone.upsert_product_chunks(preprocessed_chunks)

        return preprocessed_chunks

    async def update_product_embedding(self, product: ProductRequest) -> None:
        """
        Update the Pinecone embeddings for a product using its UID.
        
        Subject to change
        """
        print("Updating product embeddings for UID:", product.uid)

        result = await self.db.execute(
            select(ProductMinimal).where(ProductMinimal.uid == product.uid)
        )
        db_product = result.scalar_one_or_none()

        if not db_product:
            raise Exception("Product not found")

        self.pinecone.delete_by_product(self.shop_id, db_product.id)
        product.id = db_product.id

        preprocessed_chunks = preprocess_product(product, self.shop_id)
        self.pinecone.upsert_product_chunks(preprocessed_chunks)
        
        print("Updated product embeddings for UID:", product.uid)
        
        
    async def delete_product_embedding(self, product_uid: str) -> None:

        print("Deleting product embeddings for UID:", product_uid)

        result = await self.db.execute(
            select(ProductMinimal).where(ProductMinimal.uid == product_uid)
        )
        db_product = result.scalar_one_or_none()

        if not db_product:
            raise Exception("Product not found")

        self.pinecone.delete_by_product(self.shop_id, db_product.id)
        
        await self.db.delete(db_product)
        await self.db.commit()
        
        print("Deleted product embeddings for UID:", product_uid)
