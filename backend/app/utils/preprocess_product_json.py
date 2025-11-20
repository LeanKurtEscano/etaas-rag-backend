from typing import List, Dict
from schemas.product import Product  
from backend.app.rag.chunking.chunking import recursive_character_base_chunking

def preprocess_product(product: Product, store_id: str, chunk_size: int = 400) -> List[Dict]:
    """
    Convert a Product object into chunked embeddings payloads for Pinecone.
    
    Args:
        product (Product): Pydantic product object.
        store_id (str): Store ID for metadata.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to 400.
    
    Returns:
        List[Dict]: List of dicts formatted for `upsert_product_chunks`:
            [
                {
                    "text": "...",
                    "metadata": {
                        "store_id": str,
                        "product_id": str,
                        "chunk_index": int,
                        "product_name": str,
                        "category": str,
                        "base_price": float,
                        "has_variants": bool,
                        "variant_summary": str | None
                    }
                }
            ]
    """
    
    product_id = product.id
    name = product.name
    description = product.description
    category = product.category
    base_price = product.price
    has_variants = product.hasVariants

    variant_summary = None
    if has_variants and product.variants:
        summary_lines = []
        for variant in product.variants:
            combo = ", ".join(variant.combination)
            summary_lines.append(f"Variant: {combo} | Price: {variant.price} | Stock: {variant.stock}")
        variant_summary = "\n".join(summary_lines)
    
    full_text = f"""
    Product: {name}
    Category: {category}
    Base Price: {base_price}
    Has Variants: {has_variants}

    Description:
    {description}

    {variant_summary if variant_summary else ""}
    """.strip()

    chunked_data = recursive_character_base_chunking(full_text, chunk_size=chunk_size, chunk_overlap=100)

    chunks = []
    for chunk_data in chunked_data:
        chunks.append({
            "text": chunk_data["text"],
            "metadata": {
                "store_id": store_id,
                "product_id": product_id,
                "chunk_index": chunk_data["index"], 
                "product_name": name,
                "category": category,
                "base_price": base_price,
                "has_variants": has_variants,
                "variant_summary": variant_summary if chunk_data["index"] == 0 else None
            }
        })

    return chunks
