from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.core.db import Base

class Seller(Base):
    __tablename__ = "sellers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    businessName = Column(String, nullable=False)
    shopName = Column(String, nullable=False)
    addressLocation = Column(String, nullable=False)
    addressOfOwner = Column(String, nullable=False)
    contactNumber = Column(String, nullable=False)
    email = Column(String, nullable=False)
    registeredAt = Column(DateTime(timezone=True), server_default=func.now())
