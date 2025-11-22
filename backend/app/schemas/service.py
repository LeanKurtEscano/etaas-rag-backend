from pydantic import HttpUrl
from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime

class Service(BaseModel):
    id:  Optional[int] = None
    serviceName: str
    serviceDescription: str
    businessName: str
    category: str
    priceRange: Optional[str]
    availability: bool = True
    address: Optional[str]
    ownerName: Optional[str]
    contactNumber: Optional[str]
    images: Optional[List[str]] = []
    bannerImage: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    userId: str
    facebookLink: Optional[str] = ""
    uid: Optional[str] = ""
