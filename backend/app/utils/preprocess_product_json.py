from typing import List, Dict
from app.schemas.product import Product  
from app.rag.chunking.chunking import recursive_character_base_chunking

def preprocess_product(
    product: Product,
    shop_id: str,
    chunk_size: int = 400,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Convert a Product object into structured RAG chunks suitable for upserting
    into a vector database.

    The function merges product fields (name, category, price, variants, description)
    into a single text block, runs recursive character chunking, and attaches RAG
    metadata to each chunk for shop/product isolation.

    Args:
        product (Product): Validated product schema.
        shop_id (str): Identifier for the shop that owns this product.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap characters per chunk (for text continuity).

    Returns:
        List[Dict]: A list of chunks formatted for vector DB ingestion:
            [
                {
                    "text": str,
                    "metadata": {
                        "shop_id": str,
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
            summary_lines.append(
                f"Variant: {combo} | Price: {variant.price} | Stock: {variant.stock}"
            )
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
