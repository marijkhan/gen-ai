import re
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- Configuration ---
CAT_FACTS_PATH = r"D:\Work\gen-ai\tasks\week-2\day-1\cat-facts.txt"
CHROMA_PATH = r"D:\Work\gen-ai\tasks\week-2\day-1\storage"
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --- Load the document ---
with open(CAT_FACTS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded document: {len(text)} characters\n")


# ============================================================
# Strategy 1: Fixed-size chunking (no overlap)
# ============================================================
def fixed_size_chunk(text, chunk_size):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size
    return chunks


# ============================================================
# Strategy 2: Overlapping chunking (fixed-size with overlap)
# ============================================================
def overlapping_chunk(text, chunk_size, overlap):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ============================================================
# Strategy 3: Recursive chunking
#   Tries to split on paragraph boundaries first, then sentence
#   boundaries, then falls back to character-level splits.
#   Each final chunk is <= chunk_size characters.
# ============================================================
SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_chunk(text, chunk_size, separators=None):
    if separators is None:
        separators = list(SEPARATORS)

    # Base case: text fits in one chunk
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Try each separator in order of preference
    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue  # this separator doesn't split the text, try next

        # Greedily merge parts into chunks up to chunk_size
        chunks = []
        current = ""
        remaining_seps = separators[separators.index(sep) + 1:]

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # If a single part exceeds chunk_size, recurse with finer separators
                if len(part) > chunk_size:
                    chunks.extend(recursive_chunk(part, chunk_size, remaining_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c]

    # Last resort: hard character split
    return [text[i:i + chunk_size].strip()
            for i in range(0, len(text), chunk_size)
            if text[i:i + chunk_size].strip()]


# --- Build all three chunk sets ---
STRATEGIES = {
    "fixed": {
        "collection": "cat-facts-fixed",
        "chunks": fixed_size_chunk(text, CHUNK_SIZE),
        "desc": f"Fixed-size ({CHUNK_SIZE} chars, no overlap)",
    },
    "overlapping": {
        "collection": "cat-facts-overlapping",
        "chunks": overlapping_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP),
        "desc": f"Overlapping ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} char overlap)",
    },
    "recursive": {
        "collection": "cat-facts-recursive",
        "chunks": recursive_chunk(text, CHUNK_SIZE),
        "desc": f"Recursive (paragraph > sentence > word, max {CHUNK_SIZE} chars)",
    },
}

# --- Print chunk summaries ---
for name, info in STRATEGIES.items():
    chunks = info["chunks"]
    sizes = [len(c) for c in chunks]
    print(f"[{name}] {info['desc']}")
    print(f"  Chunks: {len(chunks)}  |  Avg: {sum(sizes)/len(sizes):.0f}  |  Min: {min(sizes)}  |  Max: {max(sizes)} chars")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:80].replace("\n", " ")
        print(f"    [{i}] {len(chunk)} chars: {preview}...")
    print()


# --- Embed and store each strategy in its own ChromaDB collection ---
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

for name, info in STRATEGIES.items():
    col_name = info["collection"]
    chunks = info["chunks"]

    # Clean slate
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=ollama_ef,
    )

    print(f"Embedding [{name}] -> {col_name} ({len(chunks)} chunks)...")

    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [f"{name}_{i + j}" for j in range(len(batch))]
        metadatas = [{"strategy": name, "chunk_index": i + j, "chunk_size": len(c)} for j, c in enumerate(batch)]
        collection.upsert(documents=batch, metadatas=metadatas, ids=ids)
        print(f"  Batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}")

    print(f"  Stored {collection.count()} embeddings.\n")

print("All three indexes built successfully.")
