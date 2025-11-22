from typing import List, Dict
from app.schemas.service import Service

from app.rag.embeddings.embedding import GeminiEmbedder
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.utils.preprocess_services_json import  preprocess_service


class ServiceIngestor():

    def __init__(self, shop_id: str):
        self.shop_id = shop_id
        self.pinecone = PineconeVectorStore("services-index", embedder=GeminiEmbedder())

    def preprocess_to_store_embedding(self, service: Service) -> List[Dict]:
        preproccessed_chunks = preprocess_service(service, self.shop_id)
        self.pinecone.upsert_service_chunks(preproccessed_chunks)
    