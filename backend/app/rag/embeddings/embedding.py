from typing import List
from .base.base_embedder import BaseEmbedder
from google import genai

from dotenv import load_dotenv
import os
load_dotenv()


class GeminiEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Default embedding model is Gemini.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
            
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )

    
        return [embedding.values for embedding in result.embeddings]