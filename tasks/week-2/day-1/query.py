import json
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- Configuration ---
CHROMA_PATH = r"D:\Work\gen-ai\tasks\week-2\day-1\storage"
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
TOP_N = 3

COLLECTIONS = {
    "fixed": "cat-facts-fixed",
    "overlapping": "cat-facts-overlapping",
    "recursive": "cat-facts-recursive",
}

QUESTIONS = [
    "How long do cats sleep each day?",
    "What is the largest breed of domestic cat?",
    "Can cats see in color?",
    "How fast can a cat run?",
    "Why do cats purr?",
]

# --- Setup ---
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collections = {}
for strategy, col_name in COLLECTIONS.items():
    collections[strategy] = client.get_collection(name=col_name, embedding_function=ollama_ef)
    print(f"Loaded [{strategy}] collection: {collections[strategy].count()} chunks")

print()


# --- Retrieve ---
def retrieve(collection, query, top_n=TOP_N):
    results = collection.query(query_texts=[query], n_results=top_n, include=["documents", "distances"])
    return list(zip(results["documents"][0], results["distances"][0]))


# --- Generate ---
def generate(query, context_chunks):
    context = "\n\n".join(f"- {doc}" for doc, _ in context_chunks)
    system_prompt = (
        "You are a helpful chatbot. Use only the following context to answer the question. "
        "Don't make up any new information. Keep your answer to 2-3 sentences.\n\n"
        f"Context:\n{context}"
    )
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"].strip()


# --- Run comparison ---
all_results = []

for q_idx, question in enumerate(QUESTIONS):
    print(f"{'='*80}")
    print(f"Q{q_idx+1}: {question}")
    print(f"{'='*80}")

    question_result = {"question": question, "strategies": {}}

    for strategy, collection in collections.items():
        retrieved = retrieve(collection, question)
        answer = generate(question, retrieved)

        question_result["strategies"][strategy] = {
            "chunks": [{"text": doc[:200], "distance": round(dist, 4)} for doc, dist in retrieved],
            "answer": answer,
        }

        print(f"\n  [{strategy}]")
        for i, (doc, dist) in enumerate(retrieved):
            preview = doc[:120].replace("\n", " ")
            print(f"    Chunk {i+1} (dist={dist:.4f}): {preview}...")
        print(f"    Answer: {answer}")

    all_results.append(question_result)
    print()

# --- Save results to JSON ---
output_path = r"D:\Work\gen-ai\tasks\week-2\day-1\comparison_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_path}")
