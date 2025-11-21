from typing import List, Dict
from schemas.service import Service  
from backend.app.rag.chunking.chunking import recursive_character_base_chunking

def preprocess_service(
    service: Service,
    shop_id: str,
    chunk_size: int = 400,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Convert a Service object into structured RAG chunks suitable for upserting
    into a vector database.

    The function merges service fields (name, business, category, description, etc.)
    into a single text block, runs recursive character chunking, and attaches RAG
    metadata to each chunk for shop/service isolation.

    Args:
        service (Service): Validated service schema.
        shop_id (str): Identifier for the shop that owns this service.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap characters per chunk (for text continuity).

    Returns:
        List[Dict]: A list of chunks formatted for vector DB ingestion:
            [
                {
                    "text": str,
                    "metadata": {
                        "shop_id": str,
                        "service_id": str,
                        "chunk_index": int,
                        "service_name": str,
                        "business_name": str,
                        "category": str,
                        "price_range": str | None,
                        "address": str | None,
                        "availability": bool,
                        "contact_number": str | None,
                        "banner_image": str | None
                    }
                }
            ]
    """

    service_id = service.id
    service_name = service.serviceName
    business_name = service.businessName
    category = service.category
    description = service.serviceDescription
    price_range = service.priceRange
    address = service.address
    availability = service.availability
    contact_number = service.contactNumber
    banner_image = service.bannerImage

    full_text = f"""
    Service: {service_name}
    Business: {business_name}
    Category: {category}
    Price Range: {price_range if price_range else "N/A"}
    Availability: {availability}

    Description:
    {description}
    """.strip()

    chunked_data = recursive_character_base_chunking(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for chunk_data in chunked_data:
        chunks.append({
            "text": chunk_data["text"],
            "metadata": {
                "shop_id": shop_id,
                "service_id": service_id,
                "chunk_index": chunk_data["index"],
                "service_name": service_name,
                "business_name": business_name,
                "category": category,
                "price_range": price_range,
                "address": address,
                "availability": availability,
                "contact_number": contact_number,
                "banner_image": banner_image
            }
        })

    return chunks
