class BaseLLMClient:
    """
    Base interface for any LLM client.
    Enforces a consistent method to generate text for different LLM providers.
    """
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method")