from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

class PineconeVectorStore:
    def __init__(self, index_name: str, embedder, dimension: int = 1536):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embedder = embedder
        self.index_name = index_name

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(index_name)

    def upsert_product_chunks(self, chunks: List[Dict]):
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)

        payload = []
        for vec, chunk in zip(vectors, chunks):
            _id = f"{chunk['metadata']['product_id']}_{chunk['metadata']['chunk_index']}"
            payload.append({
                "id": _id,
                "values": vec,
              "metadata": chunk["metadata"]

            })

        self.index.upsert(vectors=payload)

    def delete_by_product(self, shop_id: int, product_id: int):
        self.index.delete(
            filter={
                "shop_id": shop_id,
                "product_id": product_id
            }
        )

    def upsert_service_chunks(self, chunks: List[Dict]):
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)

        payload = []
        for vec, chunk in zip(vectors, chunks):
            _id = f"{chunk['metadata']['service_id']}_{chunk['metadata']['chunk_index']}"
            payload.append({
                "id": _id,
                "values": vec,
                 "metadata": chunk["metadata"]

            })

        self.index.upsert(vectors=payload)

    def delete_by_service(self, shop_id: int, service_id: int):
        self.index.delete(
            filter={
                "shop_id": shop_id,
                "service_id": service_id
            }
        )

    def delete_by_shop(self, shop_id: int):
        self.index.delete(
            filter={"shop_id": shop_id}
        )

    def query(self, query_text: str, shop_id: int, top_k: int = 5):
        q_embed = self.embedder.embed([query_text])[0]

        results = self.index.query(
            vector=q_embed,
            top_k=top_k,
            include_metadata=True,
            filter={
                "shop_id": shop_id
            }
        )

        return results
