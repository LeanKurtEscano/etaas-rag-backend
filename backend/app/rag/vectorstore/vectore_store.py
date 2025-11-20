from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, embedder, dimension: int = 1536):
        self.pc = Pinecone(api_key=api_key)
        self.embedder = embedder
        self.index_name = index_name

        # Create index if missing
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
        """
        chunks = [
          {
            "text": "...",
            "metadata": {
                "store_id": 1,
                "product_id": 20,
                "chunk_index": 0,
                ...
            }
          }
        ]
        """
        
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
        
        
    def delete_by_product(self, store_id: int, product_id: int):
        """
        Delete all vectors where store_id & product_id match.
        """
        self.index.delete(
            filter={
                "store_id": store_id,
                "product_id": product_id
            }
        )   
        
    def upsert_service_chunks(self, chunks: List[Dict]):
        """
        chunks = [
          {
            "text": "...",
            "metadata": {
                "store_id": 1,
                "service_id": 20,
                "chunk_index": 0,
                ...
            }
          }
          """
          
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
        


        
        
    def delete_by_service(self, store_id: int, service_id: int):
        """
        Delete all vectors where store_id & service_id match.
        """
        self.index.delete(
            filter={
                "store_id": store_id,
                "service_id": service_id
            }
        )


    def delete_by_store(self, store_id: int):
        self.index.delete(
            filter={"store_id": store_id}
        )


    def query(self, query_text: str, store_id: int, top_k: int = 5):
        """
        Retrieve product chunks, restricted by store_id.
        """
        q_embed = self.embedder.embed([query_text])[0]

        results = self.index.query(
            vector=q_embed,
            top_k=top_k,
            include_metadata=True,
            filter={
                "store_id": store_id
            }
        )

        return results
