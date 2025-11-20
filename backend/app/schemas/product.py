from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime

class VariantCombination(BaseModel):
    combination: List[str] 
    id: str
    image: Optional[str]
    price: float
    stock: int

class VariantCategory(BaseModel):
    id: str
    name: str 
    values: List[str]

class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str
    price: float
    quantity: int
    availability: Union[str, bool]  
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    images: Optional[List[str]] = []
    hasVariants: bool = False
    variantCategories: Optional[List[VariantCategory]] = []
    variants: Optional[List[VariantCombination]] = []
    sellerId: str
