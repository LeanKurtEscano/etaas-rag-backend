from pydantic import HttpUrl
from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime

class Service(BaseModel):
    id: str
    serviceName: str
    serviceDescription: str
    businessName: str
    category: str
    priceRange: Optional[str]
    availability: bool = True
    address: Optional[str]
    ownerName: Optional[str]
    contactNumber: Optional[str]
    images: Optional[List[HttpUrl]] = []
    bannerImage: Optional[HttpUrl] = None
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    userId: str
    facebookLink: Optional[str] = ""
