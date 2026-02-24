import os

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5

FAISS_INDEX_DIR = "data/faiss_index"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TRAVEL_URLS = [
    {"url": "https://www.lonelyplanet.com/germany/berlin/attractions",                "city": "berlin",    "category": "sightseeing", "price_level": "cheap"},
    {"url": "https://www.lonelyplanet.com/france/paris/attractions",                  "city": "paris",     "category": "sightseeing", "price_level": "expensive"},
    {"url": "https://www.lonelyplanet.com/spain/barcelona/attractions",               "city": "barcelona", "category": "sightseeing", "price_level": "medium"},
    {"url": "https://www.lonelyplanet.com/japan/tokyo/attractions",                   "city": "tokyo",     "category": "sightseeing", "price_level": "medium"},
    {"url": "https://www.lonelyplanet.com/articles/chicago-day-trips",          "city": "chicago",     "category": "sightseeing",         "price_level": "cheap"},
]
