import fitz
import re
import os

# --- Configuration ---
PDF_DIR = r"D:\Work\\gen-ai\\tasks\\week-1\\day-3\\input"
PDFS = {
    "constitution_pak.pdf": {
        "path": os.path.join(PDF_DIR, "constitution_pak.pdf"),
        "content_start_page": 22,  # Preamble starts here; TOC before this
    },
    "constitution_india.pdf": {
        "path": os.path.join(PDF_DIR, "constitution_india.pdf"),
        "content_start_page": 31,  # Preamble starts here; TOC before this
    },
}
MAX_CHUNK_SIZE = 1000  # ~250 tokens, well within bge-base-en-v1.5's 512-token limit


# --- Step 1: Extract text from PDFs (skip TOC pages) ---
def extract_text(pdf_path, start_page=0):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page, len(doc)):
        text += doc[page_num].get_text() + "\n"
    doc.close()
    return text


# --- Step 2: Clean extracted text ---
def clean_text(text, source):
    # Remove headers
    if "india" in source.lower():
        text = re.sub(r"^THE CONSTITUTION OF\s+INDIA\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\(Part\s+[IVXLC]+[A-Z]?\..*?\)\s*$", "", text, flags=re.MULTILINE)
    elif "pak" in source.lower():
        text = re.sub(r"^\s*CONSTITUTION OF PAKISTAN\s*\d*\s*$", "", text, flags=re.MULTILINE)

    # Remove footnote separator lines (underscores or dashes)
    text = re.sub(r"^[_]{10,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[–\-]{10,}\s*$", "", text, flags=re.MULTILINE)

    # Remove footnote lines (e.g., "1. Subs. by the Constitution...")
    text = re.sub(
        r"^\d{1,2}\.\s+(Ins\.|Added|Subs\.|Omitted|See |The word|Ins |Added |Subs |Rep\.).*$",
        "", text, flags=re.MULTILINE,
    )

    # Remove standalone page numbers (lines that are just a number)
    text = re.sub(r"^\s*\(\w+\)\s*$", "", text, flags=re.MULTILINE)  # roman numeral page nums like (xviii)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --- Step 3: Chunk by article boundaries ---
def chunk_by_article(text, source):
    # Pattern: line starting with a number (possibly with letter suffix like 2A, 31B)
    # followed by a period and whitespace. This matches article starts in both constitutions.
    article_pattern = re.compile(r"^(\d+[A-Z]?)\.\s", re.MULTILINE)

    chunks = []
    matches = list(article_pattern.finditer(text))

    if not matches:
        chunks.append({"text": text.strip(), "source": source, "article_id": "preamble"})
        return chunks

    # Capture text before the first article (preamble, part headings, etc.)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble and len(preamble) > 50:
            chunks.append({"text": preamble, "source": source, "article_id": "preamble"})

    # Split at each article boundary
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        article_id = match.group(1)

        # Skip tiny fragments (schedule entries, list items, etc.)
        if len(article_text) < 50:
            continue

        if len(article_text) > MAX_CHUNK_SIZE:
            sub_chunks = split_by_subclause(article_text, source, article_id)
            chunks.extend(sub_chunks)
        else:
            chunks.append({"text": article_text, "source": source, "article_id": article_id})

    return chunks


def split_by_subclause(text, source, article_id):
    # Pattern: sub-clause markers like (1), (2) at the start of a line (with optional leading whitespace)
    subclause_pattern = re.compile(r"^\s*\((\d+)\)", re.MULTILINE)
    matches = list(subclause_pattern.finditer(text))

    if len(matches) <= 1:
        return [{"text": text, "source": source, "article_id": article_id}]

    sub_chunks = []
    # Text before first sub-clause (article header / title)
    header = text[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start:end].strip()

        # Prepend header to first sub-chunk for context
        if i == 0 and header:
            sub_text = header + "\n" + sub_text

        clause_num = match.group(1)
        sub_chunks.append({
            "text": sub_text,
            "source": source,
            "article_id": f"{article_id}({clause_num})",
        })

    return sub_chunks if sub_chunks else [{"text": text, "source": source, "article_id": article_id}]


def enforce_max_size(chunks, max_size):
    """Hard-split any chunks still exceeding max_size."""
    result = []
    for chunk in chunks:
        if len(chunk["text"]) <= max_size:
            result.append(chunk)
        else:
            parts = _split_text(chunk["text"], max_size)
            for i, part_text in enumerate(parts):
                result.append({
                    "text": part_text,
                    "source": chunk["source"],
                    "article_id": f"{chunk['article_id']}_p{i+1}" if len(parts) > 1 else chunk["article_id"],
                })
    return result


def _split_text(text, max_size):
    """Split text into pieces <= max_size, preferring paragraph then line boundaries."""
    if len(text) <= max_size:
        return [text]

    # Try splitting on double newlines first, then single newlines
    for sep in ["\n\n", "\n"]:
        segments = text.split(sep)
        if len(segments) > 1:
            pieces = []
            current = ""
            for seg in segments:
                candidate = current + sep + seg if current else seg
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current.strip())
                    # If a single segment exceeds max_size, recurse with next separator
                    if len(seg) > max_size:
                        pieces.extend(_split_text(seg, max_size))
                        current = ""
                    else:
                        current = seg
            if current:
                pieces.append(current.strip())
            if pieces:
                return [p for p in pieces if p]

    # Last resort: hard split by character count
    return [text[i : i + max_size].strip() for i in range(0, len(text), max_size) if text[i : i + max_size].strip()]


# --- Step 4: Process both PDFs and print summary ---
all_chunks = []

for filename, info in PDFS.items():
    print(f"\nProcessing: {filename}")
    raw_text = extract_text(info["path"], info["content_start_page"])
    cleaned = clean_text(raw_text, filename)
    chunks = chunk_by_article(cleaned, filename)
    chunks = enforce_max_size(chunks, MAX_CHUNK_SIZE)
    all_chunks.extend(chunks)

    # Summary stats
    sizes = [len(c["text"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk size: {avg_size:.0f} chars")
    print(f"  Min/Max chunk size: {min(sizes)}/{max(sizes)} chars")

    # Print first 3 chunks
    print(f"\n  First 3 chunks from {filename}:")
    for j, chunk in enumerate(chunks[:3]):
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"    [{j}] article_id={chunk['article_id']}, size={len(chunk['text'])} chars")
        print(f"        {preview}...")
    print()

print(f"Total chunks across all PDFs: {len(all_chunks)}")


# --- Step 5: Embed and store in ChromaDB ---
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
CHROMA_PATH = r"D:\\Work\\gen-ai\\tasks\\week-1\\day-5\\storage"
COLLECTION_NAME = "constitutions"

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef,
)

existing_count = collection.count()

if existing_count >= len(all_chunks):
    print(f"\nChromaDB: Collection '{COLLECTION_NAME}' already has {existing_count} embeddings, skipping.")
else:
    if existing_count > 0:
        print(f"\nChromaDB: Collection has {existing_count} docs but expected {len(all_chunks)}, re-embedding...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ollama_ef,
        )

    print(f"\nChromaDB: Embedding {len(all_chunks)} chunks (this may take a while)...")

    # Assign unique IDs (index-based to avoid duplicates from Schedule sections)
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = f"{chunk['source']}_chunk{idx}"

    # Upsert in batches to avoid hitting Ollama/ChromaDB limits
    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        collection.upsert(
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "article_id": c["article_id"]} for c in batch],
            ids=[c["id"] for c in batch],
        )
        print(f"  Embedded batch {i // BATCH_SIZE + 1}/{(len(all_chunks) - 1) // BATCH_SIZE + 1}")

    print(f"ChromaDB: Done. Stored {collection.count()} embeddings.")


# --- Step 6: Retrieval ---
def retrieve(query, top_n=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))


# --- Step 7: Generation ---
def generate(query, context_chunks):
    context = "\n\n".join(
        f"[{meta['source']}, Article {meta['article_id']}]\n{doc}"
        for doc, meta, _ in context_chunks
    )
    system_prompt = f"""You are a constitutional law expert. Answer the question using ONLY the provided context from the constitutions of Pakistan and India. If the answer is not in the context, say so. Cite the source (Pakistan/India) and article number when possible.

Context:
{context}"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )
    print("\nAnswer: ")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")


# --- Step 8: Interactive loop ---
print("\n--- Constitutional RAG Pipeline Ready ---")
print("Ask questions about the constitutions of Pakistan and India.")
print("Type 'quit' to exit.\n")

while True:
    query = input("Question: ").strip()
    if not query or query.lower() in ("quit", "exit", "q"):
        break

    results = retrieve(query)

    print("\nRetrieved chunks:")
    for doc, meta, dist in results:
        preview = doc[:120].replace("\n", " ")
        print(f"  [{meta['source']}, Art. {meta['article_id']}] (dist={dist:.3f}) {preview}...")

    generate(query, results)
