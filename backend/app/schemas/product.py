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
    
class ProductRequest(BaseModel):
    id: Optional[int] = None
    name: str
    description: str
    category: str
    price: float
    quantity: int
    availability: Union[str, bool]
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    images: Optional[List[str]] = []
    hasVariants: bool
    variantCategories: Optional[List[VariantCategory]] = []
    variants: Optional[List[VariantCombination]] = []
    sellerId: Union[str, int]
    uid: Optional[str] = ""



class Product(BaseModel):
    id: int
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
    sellerId: Union[str, int]

class ProductMinimal(BaseModel):
    id: str
    name: str