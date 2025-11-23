from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.core.db import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)     
    shop_id = Column(Integer, index=True)
    
    role = Column(String, index=True)        
    
    message = Column(Text)                   
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
