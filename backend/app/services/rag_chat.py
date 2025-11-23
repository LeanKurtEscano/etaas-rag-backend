from app.rag.pipeline import AgenticRAGPipeline
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chats import ChatMessage  

class RAGService:
    def __init__(self, pipeline: AgenticRAGPipeline, db: AsyncSession):
        self.pipeline = pipeline
        self.db = db
        
    async def chat(self, shop_id: int, query: str, user_id: str):

     
        user_msg = ChatMessage(
            user_id=user_id,
            shop_id=shop_id,
            role="user",
            message=query
        )
        self.db.add(user_msg)
        await self.db.commit()

     
        result = await self.pipeline.run(shop_id=shop_id, query=query)
        ai_answer = result.get("answer", "")

 
        ai_msg = ChatMessage(
            user_id=user_id,
            shop_id=shop_id,
            role="assistant",
            message=ai_answer
        )
        self.db.add(ai_msg)
        await self.db.commit()

       
        return result
