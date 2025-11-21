from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_character_base_chunking(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Production-ready recursive character chunking using LangChain.
    Returns a list of dicts: [{ "text": str, "index": int }]
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    chunks = splitter.split_text(text)

    return [
        {"text": chunk, "index": i}
        for i, chunk in enumerate(chunks)
    ]
