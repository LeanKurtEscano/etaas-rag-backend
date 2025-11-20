from langchain_google_genai import GoogleGenerativeAIEmbeddings

def generate_embeddings(texts: str | list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Google Generative AI Embeddings.
    Returns a list of embeddings.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings = embedding_model.embed_documents(texts)
    return embeddings