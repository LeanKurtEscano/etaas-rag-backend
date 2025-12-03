# app/rag/agentic_pipeline.py
from typing import List
from app.rag.generation.reranker import Reranker
from app.rag.vectorstore.vectore_store import PineconeVectorStore
import logging

from app.rag.agents.index_routing_agent import IndexRoutingAgent
from app.rag.agents.response_generation_agent import ResponseGenerationAgent
logger = logging.getLogger(__name__)

class AgenticRAGPipeline:
    """
    Agentic RAG: LLM decides which index to query, retrieves results, reranks, 
    builds context, and generates a response.
    """

    def __init__(self, vectorstores: dict, llm):
        """
        vectorstores: dict of index_name -> PineconeVectorStore
        llm: synchronous LLM wrapper with .generate(prompt: str) -> str
        """
        self.vectorstores = vectorstores
        self.llm = llm
        self.reranker = Reranker()
        self.routing_agent = IndexRoutingAgent(llm)
        self.response_agent = ResponseGenerationAgent(llm)

    def retrieve(self, query: str, shop_id: int, index_name: str, top_k: int = 5) -> List[str]:
        """
        Retrieve candidate chunks from a specific index.
        """
        try:
            vs = self.vectorstores[index_name]
            results = vs.query(query, shop_id=shop_id, top_k=top_k)
            if not results or not getattr(results, "matches", []):
                return []
            return [
                match.metadata.get("text", str(match.metadata))
                for match in results.matches
            ]
        except Exception as e:
            logger.exception(f"Vectorstore retrieval failed for {index_name}: {e}")
            return []

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        try:
            ranked = self.reranker.rerank(query, candidates)
            return [c for c, _ in ranked]
        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            return candidates

    def build_context(self, top_contents: List[str], max_chars: int = 2000) -> str:
        context = ""
        for chunk in top_contents:
            if len(context) + len(chunk) > max_chars:
                break
            context += chunk + "\n\n"
        return context.strip()



    def run(self, query: str, shop_id: int, top_k: int = 5):
        """
        Agentic pipeline execution:
        1. Ask LLM which indexes to query
        2. Retrieve from selected indexes
        3. Rerank and build context
        4. Generate answer
        """
        indexes = self.routing_agent.decide_index(query)
        all_chunks = []

        for index_name in indexes:
            chunks = self.retrieve(query, shop_id=shop_id, index_name=index_name, top_k=top_k)
            all_chunks.extend(chunks)

        top_chunks = self.rerank(query, all_chunks)[:top_k]
        context = self.build_context(top_chunks)
        answer = self.response_agent.generate(query, context)
        print("answer:", answer)

        return {
            "answer": answer,
            "context_used": context,
            "indexes_queried": indexes,
            "retrieved_docs": len(top_chunks)
        }
