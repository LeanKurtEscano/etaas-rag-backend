# backend/app/rag/ingest/ingest.py
from fastapi import APIRouter, HTTPException
from typing import List

from schemas.product import Product
from schemas.service import Service
from backend.app.services.product_ingestor import ProductIngestor
from backend.app.services.services_ingestor import ServiceIngestor

router = APIRouter()

@router.post("/shops/{shop_id}/products")
async def ingest_products(shop_id: str, products: List[Product]):
    ingestor = ProductIngestor(shop_id)
    for product in products:
        ingestor.preprocess_to_store_embedding(product)
    return {"status": "success", "ingested_products": len(products)}


@router.post("/shops/{shop_id}/service")
async def ingest_service(shop_id: str, service: Service):
    ingestor = ServiceIngestor(shop_id)
    ingestor.preprocess_to_store_embedding(service)
    return {"status": "success", "ingested_service": service.id}
