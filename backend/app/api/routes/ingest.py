# backend/app/rag/ingest/ingest.py
from fastapi import APIRouter,Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db

from app.schemas.product import Product, ProductRequest
from app.schemas.service import Service
from app.services.product_ingestor import ProductIngestor
from app.services.services_ingestor import ServiceIngestor

router = APIRouter()

@router.post("/shops/{shop_id}/products")
async def ingest_products(shop_id: str, product: ProductRequest,db: AsyncSession = Depends(get_db)):
  
    ingestor = ProductIngestor(shop_id,db = db)
    await ingestor.preprocess_to_store_embedding(product)
    return {"status": "success", "ingested_products": 1}


@router.post("/shops/{shop_id}/service")
async def ingest_service(shop_id: str, service: Service):
    ingestor = ServiceIngestor(shop_id)
    await ingestor.preprocess_to_store_embedding(service)
    return {"status": "success", "ingested_service": service.id}
