
from fastapi import APIRouter, Query,Depends
from fastapi.responses import StreamingResponse
import asyncio
from app.rag.pipeline import AgenticRAGPipeline
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.rag.embeddings.embedding import GeminiEmbedder
from app.core.gemini import GeminiLLMClient
from app.services.rag_chat import RAGService
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db
from models.chats import ChatMessage
from pydantic import BaseModel

router = APIRouter()

vectorstores = {
    "products-index": PineconeVectorStore("products-index", embedder=GeminiEmbedder()),
    "services-index": PineconeVectorStore("services-index", embedder=GeminiEmbedder()),
}



llm = GeminiLLMClient()
pipeline = AgenticRAGPipeline(vectorstores, llm)
service = RAGService(pipeline)

class ChatRequest(BaseModel):
    user_id: int
    query: str
    
async def stream_answer(query: str, shop_id: int, user_id: str):
    result = await service.chat(shop_id=shop_id, query=query, user_id=user_id) 
    answer = result.get("answer", "")
    for line in answer.splitlines():
        yield f"data: {line}\n\n"
        await asyncio.sleep(0.05)



@router.post("/shops/{shop_id}/agentic-chat")
async def chat(shop_id: int, data: ChatRequest):
    return StreamingResponse(
        stream_answer(data.query, shop_id, data.user_id),
        media_type="text/event-stream"
    )
    
    
@router.get("/shops/{shop_id}/chat-history/{user_id}")
async def get_chat_history(shop_id: int, user_id: str,db: AsyncSession = Depends(get_db)):

    chats = (
        db.query(ChatMessage)
        .filter(ChatMessage.shop_id == shop_id,
                ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return chats
