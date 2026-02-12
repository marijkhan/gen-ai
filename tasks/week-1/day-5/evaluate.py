import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import json
from datetime import datetime

# --- Configuration (same as task.py) ---
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
CHROMA_PATH = r"D:\Work\gen-ai\tasks\week-1\day-5\storage"
COLLECTION_NAME = "constitutions"
OUTPUT_FILE = r"D:\Work\gen-ai\tasks\week-1\day-5\rag_responses.json"

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

# --- Connect to existing ChromaDB ---
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


def generate(query, context_chunks):
    context = "\n\n".join(
        f"[{meta['source']}, Article {meta['article_id']}]\n{doc}"
        for doc, meta, _ in context_chunks
    )
    system_prompt = f"""You are a constitutional law expert. Answer the question using ONLY the provided context from the constitutions of Pakistan and India. If the answer is not in the context, say so. Cite the source (Pakistan/India) and article number when possible.

Context:
{context}"""

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]


# --- Run all questions ---
results = []

for q in QUESTIONS:
    print(f"Q{q['id']}: {q['question']}")

    retrieved = retrieve(q["question"])
    retrieved_info = [
        {"source": meta["source"], "article_id": meta["article_id"], "distance": round(dist, 4)}
        for _, meta, dist in retrieved
    ]

    answer = generate(q["question"], retrieved)
    print(f"A{q['id']}: {answer[:200]}...\n")

    results.append({
        "id": q["id"],
        "question": q["question"],
        "ground_truth": q["ground_truth"],
        "rag_response": answer,
        "retrieved_chunks": retrieved_info,
    })

# --- Save to file ---
output = {
    "model": LANGUAGE_MODEL,
    "embedding_model": EMBEDDING_MODEL,
    "timestamp": datetime.now().isoformat(),
    "results": results,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nDone. Responses saved to {OUTPUT_FILE}")
