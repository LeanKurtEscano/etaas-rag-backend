from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pydantic import BaseModel

from app.core.db import get_db
from app.models.chats import ChatMessage

from app.rag.pipeline import AgenticRAGPipeline
from app.rag.vectorstore.vectore_store import PineconeVectorStore
from app.rag.embeddings.embedding import GeminiEmbedder
from app.core.gemini import GeminiLLMClient
from app.services.rag_chat import RAGService

router = APIRouter()


vectorstores = {
    "products-index": PineconeVectorStore("products-index", embedder=GeminiEmbedder()),
    "services-index": PineconeVectorStore("services-index", embedder=GeminiEmbedder()),
}

llm = GeminiLLMClient()
pipeline = AgenticRAGPipeline(vectorstores, llm)

class ChatRequest(BaseModel):
    user_id: str
    query: str


def get_rag_service(
    db: AsyncSession = Depends(get_db),
):
    return RAGService(pipeline=pipeline, db=db)



async def stream_answer(service: RAGService, query: str, shop_id: int, user_id: str):
    result = await service.chat(shop_id=shop_id, query=query, user_id=user_id)
    answer = result.get("answer", "")

    for line in answer.splitlines():
        yield f"data: {line}\n\n"
        await asyncio.sleep(0.05)



@router.post("/shops/{shop_id}/agentic-chat")
async def chat(
    shop_id: int,
    data: ChatRequest,
    service: RAGService = Depends(get_rag_service) 
):
    return StreamingResponse(
        stream_answer(service, data.query, shop_id, data.user_id),
        media_type="text/event-stream"
    )


@router.get("/shops/{shop_id}/chat-history/{user_id}")
async def get_chat_history(
    shop_id: int,
    user_id: str,
    db: AsyncSession = Depends(get_db)
):

    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.shop_id == shop_id,
            ChatMessage.user_id == user_id
        )
        .order_by(ChatMessage.created_at.asc())
    )

    result = await db.execute(stmt)
    chats = result.scalars().all()

    return chats
