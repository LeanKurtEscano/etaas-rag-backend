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

@router.put("/shops/{shop_id}/products")
async def update_product_embeddings(shop_id: str, product: ProductRequest,db: AsyncSession = Depends(get_db)):
    ingestor = ProductIngestor(shop_id,db = db)
    await ingestor.update_product_embedding(product)
    return {"status": "success", "updated_product_uid": product.uid}

@router.delete("/shops/{shop_id}/products/{product_uid}")
async def delete_product_embeddings(shop_id: str, product_uid: str,db: AsyncSession = Depends(get_db)):
    ingestor = ProductIngestor(shop_id,db = db)
    await ingestor.delete_product_embedding(product_uid)
    return {"status": "success", "deleted_product_uid": product_uid}


@router.post("/shops/{shop_id}/service")
async def ingest_service(shop_id: str, service: Service,db: AsyncSession = Depends(get_db)):
    ingestor = ServiceIngestor(shop_id,db = db)
    await ingestor.preprocess_to_store_embedding(service)
    return {"status": "success", "ingested_service": service.id}


@router.put("/shops/{shop_id}/service")
async def update_service_embeddings(shop_id: str, service: Service,db: AsyncSession = Depends(get_db)):
    ingestor = ServiceIngestor(shop_id,db = db)
    await ingestor.update_service_embedding(service)
    return {"status": "success", "updated_service_uid": service.uid}

@router.delete("/shops/{shop_id}/service/{service_uid}")
async def delete_service_embeddings(shop_id: str, service_uid: str,db: AsyncSession = Depends(get_db)):
    ingestor = ServiceIngestor(shop_id,db = db)
    await ingestor.delete_service_embedding(service_uid)
    return {"status": "success", "deleted_service_uid": service_uid}