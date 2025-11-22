from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.core.db import Base

class ServiceMinimal(Base):
    __tablename__ = "services_minimal"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    uid = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
