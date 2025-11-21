from typing import List, Dict
from schemas.service import Service  
from backend.app.rag.chunking.chunking import recursive_character_base_chunking
def preprocess_service_json(service: Service, store_id: str, chunk_size: int = 400) -> List[Dict]:
    """
    Convert a Service object into chunked embeddings payloads for Pinecone.
    
    Args:
        service (Service): Pydantic service object.
        store_id (str): Store ID for metadata.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to 400.
    
    Returns:
        List[Dict]: List of dicts formatted for `upsert_service_chunks`:
            [
                {
                    "text": "...",
                    "metadata": {
                        "store_id": str,
                        "service_id": str,
                        "chunk_index": int,
                        "service_name": str,
                        "business_name": str,
                        "category": str,
                        "price_range": str | None,
                        "address": str | None,
                        "availability": bool,
                        "contact_number": str | None,
                        "banner_image": str | None,
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

 
    chunked_data = recursive_character_base_chunking(full_text, chunk_size=400, chunk_overlap=100)
    
    chunks = []
    for chunk_data in chunked_data:
        chunks.append({
            "text": chunk_data["text"],
            "metadata": {
                "store_id": store_id,
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
