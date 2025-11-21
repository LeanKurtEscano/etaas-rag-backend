# backend/app/rag/embeddings/gemini_embedder.py

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .base.base_embedder import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "models/gemini-embedding-001"):
        """
        Default embedding model is Gemini.
        """
        self.model = GoogleGenerativeAIEmbeddings(model=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
 
        if isinstance(texts, str):
            texts = [texts]

        return self.model.embed_query(texts)
