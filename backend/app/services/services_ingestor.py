from sqlalchemy import select
from typing import List, Dict
from app.schemas.service import Service
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag.embeddings.embedding import GeminiEmbedder
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.utils.preprocess_services_json import  preprocess_service
from app.models.service import ServiceMinimal

class ServiceIngestor():

    def __init__(self, shop_id: int,db: AsyncSession):
        self.shop_id = shop_id
        self.db = db
        self.pinecone = PineconeVectorStore("services-index", embedder=GeminiEmbedder(),dimension=3072)

    async def preprocess_to_store_embedding(self, service: Service) -> List[Dict]:
        
        db_service = ServiceMinimal(name=service.serviceName, uid = service.uid)
        self.db.add(db_service)
        await self.db.commit()        
        await self.db.refresh(db_service) 
        
       
        service.id = db_service.id
        preproccessed_chunks = preprocess_service(service, self.shop_id)
        self.pinecone.upsert_service_chunks(preproccessed_chunks)
        
        
    async def update_service_embedding(self, service: Service) -> None:
        """
        Update the Pinecone embeddings for a service using its UID.
        
        Subject to change
        """
        print("Updating service embeddings for UID:", service.uid)

        query = select(ServiceMinimal).where(ServiceMinimal.uid == service.uid)
        result = await self.db.execute(query)
        db_service = result.scalar_one_or_none()
        

        if not db_service:
            raise Exception("Service not found")

        self.pinecone.delete_by_service(self.shop_id, db_service.id)
        service.id = db_service.id
        preprocessed_chunks = preprocess_service(service, self.shop_id)
        self.pinecone.upsert_service_chunks(preprocessed_chunks)
        
        print("Updated service embeddings for UID:", service.uid)
        
        
    async def delete_service_embedding(self, service_uid: str) -> None:

        print("Deleting service embeddings for UID:", service_uid)

        result = await self.db.execute(
            select(ServiceMinimal).where(ServiceMinimal.uid == service_uid)
        )
        db_service = result.scalar_one_or_none()

        if not db_service:
            raise Exception("Service not found")

        self.pinecone.delete_by_service(self.shop_id, db_service.id)
        
        await self.db.delete(db_service)
        await self.db.commit()
        
        print("Deleted service embeddings for UID:", service_uid)

        
    
    