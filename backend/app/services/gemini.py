from google import genai

from dotenv import load_dotenv
import os

load_dotenv()

def initialize_gemini_client():
    """
    Initialize and return a Gemini client with the provided API key.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client