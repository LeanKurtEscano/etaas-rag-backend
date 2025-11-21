import google.generativeai as genai
from dotenv import load_dotenv
import os
from backend.app.rag.generation.base.base_generator import BaseLLMClient
load_dotenv()



class GeminiLLMClient(BaseLLMClient):
    """
    Gemini LLM client using Google GenAI.
    Handles initialization and text generation.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Gemini client.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini LLM.

        Args:
            prompt (str): The input prompt to generate text from.
            **kwargs: Additional parameters for the model.

        Returns:
            str: Generated text from the model.
        """
        response = self.client.generate_content(model="gemini-2.5-flash", contexts=prompt, **kwargs)
        return response.text