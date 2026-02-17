import fitz
import re
import os
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import json
from datetime import datetime

# --- Configuration ---
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
CHROMA_PATH = r"D:\Work\gen-ai\tasks\week-1\day-5\storage"
COLLECTION_NAME = "constitutions"
OUTPUT_FILE = r"D:\Work\gen-ai\tasks\week-1\day-5\rag_comparison.json"
PDF_DIR = r"D:\Work\gen-ai\tasks\week-1\day-3\input"

# Max chars per PDF for direct context (~4K tokens each, ~8K total)
MAX_CONTEXT_CHARS_PER_PDF = 15000
# Ollama context window to request
NUM_CTX = 8192

QUESTIONS = [
    {
        "id": 1,
        "question": "What is the minimum age requirement to become the President of Pakistan as stated in the Constitution?",
        "ground_truth": "45 years. Article 41, Clause (2)",
    },
    {
        "id": 2,
        "question": "How many articles are in the Constitution of India?",
        "ground_truth": "395",
    },
    {
        "id": 3,
        "question": "What language is specified as the national language of Pakistan?",
        "ground_truth": "Urdu",
    },
    {
        "id": 4,
        "question": "What is the maximum term length of the Indian Lok Sabha?",
        "ground_truth": "5 years",
    },
    {
        "id": 5,
        "question": "Which article of Pakistan's Constitution deals with the abolition of exploitation?",
        "ground_truth": "Article 3",
    },
    {
        "id": 6,
        "question": "At what age does a Judge of the Supreme Court for both countries retire?",
        "ground_truth": "65 years",
    },
    {
        "id": 7,
        "question": "Who is the highest authority in each government?",
        "ground_truth": "Prime minister",
    },
    {
        "id": 8,
        "question": "Who wrote the preface for the constitution?",
        "ground_truth": "Dr. Reeta Vasishta wrote the preface for the Indian constitution. Raja Naeem Akbar wrote the preface for the Pakistan constitution",
    },
    {
        "id": 9,
        "question": "Which article covers the freedom of speech?",
        "ground_truth": "Article 19",
    },
    {
        "id": 10,
        "question": "How many amendments has there been in each constitution?",
        "ground_truth": "Not explicitly stated",
    },
    {
        "id": 11,
        "question": "What did the 1986 amendment in Pakistan constitution state?",
        "ground_truth": "No such amendment exists",
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are a constitutional law expert. Answer the question using ONLY the provided context from the constitutions of Pakistan and India. If the answer is not in the context, say so. Cite the source (Pakistan/India) and article number when possible.

Context:
{context}"""


# ============================
# Part 1: RAG-based responses
# ============================
print("=" * 60)
print("PART 1: RAG-based responses")
print("=" * 60)

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef,
)
print(f"ChromaDB: Loaded collection with {collection.count()} embeddings.\n")


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


def generate_rag(query, context_chunks):
    context = "\n\n".join(
        f"[{meta['source']}, Article {meta['article_id']}]\n{doc}"
        for doc, meta, _ in context_chunks
    )
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]


rag_results = {}
for q in QUESTIONS:
    print(f"  Q{q['id']}: {q['question']}")
    retrieved = retrieve(q["question"])
    answer = generate_rag(q["question"], retrieved)
    print(f"  A{q['id']}: {answer[:150]}...\n")
    rag_results[q["id"]] = {
        "response": answer,
        "retrieved_chunks": [
            {"source": meta["source"], "article_id": meta["article_id"], "distance": round(dist, 4)}
            for _, meta, dist in retrieved
        ],
    }


# ==========================================
# Part 2: Direct LLM responses (no RAG)
# ==========================================
print("=" * 60)
print("PART 2: Direct LLM responses (full PDF text as context)")
print("=" * 60)

PDFS = [
    ("Constitution of Pakistan", os.path.join(PDF_DIR, "constitution_pak.pdf"), 22),
    ("Constitution of India", os.path.join(PDF_DIR, "constitution_india.pdf"), 31),
]


def extract_and_clean(pdf_path, start_page):
    doc = fitz.open(pdf_path)
    text = ""
    for p in range(start_page, len(doc)):
        text += doc[p].get_text() + "\n"
    doc.close()
    # Light cleaning
    text = re.sub(r"^[_]{10,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[–\-]{10,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Build the direct context (truncated to fit model context window)
direct_context_parts = []
for label, path, start in PDFS:
    full_text = extract_and_clean(path, start)
    truncated = full_text[:MAX_CONTEXT_CHARS_PER_PDF]
    direct_context_parts.append(f"--- {label} ---\n{truncated}")
    print(f"  {label}: {len(full_text):,} chars total, using first {len(truncated):,} chars")

direct_context = "\n\n".join(direct_context_parts)
print(f"  Combined context: {len(direct_context):,} chars (~{len(direct_context) // 4:,} tokens)\n")


def generate_direct(query, context):
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
            {"role": "user", "content": query},
        ],
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"]


direct_results = {}
for q in QUESTIONS:
    print(f"  Q{q['id']}: {q['question']}")
    answer = generate_direct(q["question"], direct_context)
    print(f"  A{q['id']}: {answer[:150]}...\n")
    direct_results[q["id"]] = {"response": answer}


# ==========================================
# Part 3: Build comparison output
# ==========================================
comparison = {
    "model": LANGUAGE_MODEL,
    "embedding_model": EMBEDDING_MODEL,
    "timestamp": datetime.now().isoformat(),
    "notes": {
        "rag": "Top 5 chunks retrieved from ChromaDB using bge-base-en-v1.5 embeddings",
        "direct": f"First {MAX_CONTEXT_CHARS_PER_PDF:,} chars per PDF passed as context with num_ctx={NUM_CTX}",
    },
    "results": [],
}

for q in QUESTIONS:
    comparison["results"].append({
        "id": q["id"],
        "question": q["question"],
        "ground_truth": q["ground_truth"],
        "rag_response": rag_results[q["id"]]["response"],
        "direct_response": direct_results[q["id"]]["response"],
        "retrieved_chunks": rag_results[q["id"]]["retrieved_chunks"],
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print("=" * 60)
print(f"Done. Comparison saved to {OUTPUT_FILE}")
print("=" * 60)
