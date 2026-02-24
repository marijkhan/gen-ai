# Preference-Aware Travel RAG Assistant

A complete end-to-end Retrieval-Augmented Generation (RAG) system for travel recommendations, built with Streamlit.

## Setup

```bash
pip install -r requirements.txt
```

Create `assignment1/.env`:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
streamlit run assignment1/app.py
```

## How It Works

### Pipeline (7 Steps)
1. **Extract preferences** — Groq LLM parses the query into `{city, budget, interests}`
2. **Embed query** — SentenceTransformer encodes query to a 384-dim vector
3. **Semantic search** — FAISS retrieves top-10 candidate chunks via cosine similarity
4. **Metadata filter** — Hard-filter by city, price level, and category
5. **Re-rank** — Composite score `0.6 × semantic + 0.4 × preference_bonus`
6. **Judge context** — LLM checks if retrieved context is sufficient; relaxes filters if not
7. **Generate answer** — LLM produces a cited recommendation grounded in retrieved chunks

### Design Choices

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Lightweight, CPU-friendly, 384-dim vectors — no GPU required |
| Vector database | FAISS `IndexFlatIP` | No server needed; single binary file; exact cosine similarity with L2-normalised vectors |
| LLM | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq | Fast inference, strong instruction following |
| Re-ranking | Score-based composite | Avoids an extra LLM call; transparent and deterministic |
| Chunking | Paragraph-first, 800 chars | Travel content is naturally paragraph-structured |

## Knowledge Base

10 pages across 4 cities: **Berlin**, **Paris**, **Barcelona**, **Tokyo**
Sources: Lonely Planet (attractions) and Timeout (restaurants, art galleries)

The FAISS index is built on first run and persisted to `data/faiss_index/`. Subsequent runs load from disk.

## Failure Case

If you query for a city not in the knowledge base (e.g. Amsterdam), the pipeline:
1. Retrieves the closest chunks by semantic similarity
2. Judges the context as insufficient
3. Relaxes filters to city-only and retries
4. Returns a polite refusal message if context is still insufficient

Example refusal query: *"Best restaurants in Amsterdam"*
