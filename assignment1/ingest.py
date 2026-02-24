import json
import logging
import os

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

from config import CHUNK_OVERLAP, CHUNK_SIZE, FAISS_INDEX_DIR, REQUEST_HEADERS, TRAVEL_URLS
from embedder import embed_texts
from chonkie import RecursiveChunker

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 200


def fetch_url(url: str, headers: dict) -> str:
    """Fetch HTML from a URL. Returns empty string on failure."""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def clean_html(html: str) -> str:
    """Remove boilerplate, extract meaningful text from paragraphs and list items."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    parts = []
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 30:
            parts.append(text)
    return "\n\n".join(parts)

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    chunker = RecursiveChunker()
    chunks = chunker(text)
    return [chunk.text for chunk in chunks]
    


def build_index(
    urls: list[dict] = TRAVEL_URLS
):
    """Fetch, clean, chunk, embed all URLs and build a FAISS IndexFlatIP."""
    all_chunks = []
    all_metadata = []

    for entry in urls:
        url = entry["url"]
        logger.info(f"Fetching: {url}")
        html = fetch_url(url, REQUEST_HEADERS)
        if not html:
            logger.warning(f"Skipping {url}: empty response")
            continue

        logger.info(f"\nRaw content: \n{html} \n\n")
        
        text = clean_html(html)
        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(f"Skipping {url}: insufficient text ({len(text)} chars)")
            continue

        logger.info(f"\nCleaned content: \n{text} \n\n")

        chunks = chunk_text(text)
        logger.info(f"  → {len(chunks)} chunks from {url}")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "url": url,
                "city": entry["city"],
                "category": entry["category"],
                "price_level": entry["price_level"],
                "text": chunk,
            })

    if not all_chunks:
        raise RuntimeError("No content could be fetched from any URL.")

    logger.info(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    return index, all_metadata


def save_index(index: faiss.Index, metadata: list[dict], path: str = FAISS_INDEX_DIR) -> None:
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Index saved to {path}")


def load_index(path: str = FAISS_INDEX_DIR):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Index loaded from {path}: {index.ntotal} vectors")
    return index, metadata


def index_exists(path: str = FAISS_INDEX_DIR) -> bool:
    return (
        os.path.exists(os.path.join(path, "index.faiss"))
        and os.path.exists(os.path.join(path, "metadata.json"))
    )
