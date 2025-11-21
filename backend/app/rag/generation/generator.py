from typing import Any

class LLMShoppingAssistant:
    """
    Production-ready AI shopping assistant for a multi-store e-commerce platform.

    Responsibilities:
    - Generate responses strictly based on retrieved RAG context.
    - Enforce rules for handling unknown products/services.
    - Dynamically supports any LLM client.
    """

    DEFAULT_SYSTEM_PROMPT = """
    You are an AI shopping assistant for an e-commerce platform. 
    Each store has its own private product catalog, stored in a vector database and retrieved using RAG.
    Only use the context retrieved from RAG.
    
    Rules:
    1. Only answer based on the retrieved context.
    2. If a product/service is not found, respond: "This store currently doesn’t offer that product/service."
    3. Never mention embeddings, vector databases, or internal system processes.
    4. Respond clearly, concisely, and helpfully.
    5. For recommendations, only recommend items present in the context.
    6. Use product attributes (price, description, variants, availability, category) accurately.
    7. If no context is retrieved, respond: "I don’t have information available for that product right now."
    8. Do not hallucinate or invent items, prices, or features.
    """

    def __init__(self, llm_client: Any):
        """
        Initialize the assistant.

        Args:
            llm_client: An LLM client instance (e.g., Gemini, OpenAI). Must support `generate()` or similar.
        """
        self.llm_client = llm_client
        

    def _build_prompt(self, user_query: str, context: str) -> str:
        """
        Construct the final LLM prompt including system rules, RAG context, and user query.
        """
        prompt = f"""
        {self.DEFAULT_SYSTEM_PROMPT}

        Context (RAG Output):
        {context}

        User Query:
        {user_query}
        """
        return prompt.strip()
    
    

    def generate_response(self, user_query: str, context: str, **kwargs) -> str:
        """
        Generate a response for a user query given retrieved RAG context.

        Args:
            user_query: User's input query.
            context: Retrieved RAG context from the vector store.
            **kwargs: Optional parameters to pass to the LLM client.

        Returns:
            str: LLM-generated response.
        """
        prompt = self._build_prompt(user_query, context)
        response = self.llm_client.generate(prompt, **kwargs)  # adapt depending on LLM SDK
        return response
