# Architecture Diagram — Preference-Aware Travel RAG Assistant

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI  (app.py)                             │
│                                                                             │
│  ┌───────────────────────────────────┐   ┌───────────────────────────────┐  │
│  │  Query Input                      │   │  Debug Panel (expandable)     │  │
│  │  "nature places in chicago"       │   │  • Extracted Preferences JSON │  │
│  │  [Find Recommendations]           │   │  • Context Verdict            │  │
│  └──────────────┬────────────────────┘   │  • URLs Used                  │  │
│                 │                         │  • Top Chunks + Scores        │  │
│  ┌──────────────▼────────────────────┐   └───────────────────────────────┘  │
│  │  Answer Display                   │                                      │
│  │  "If you're looking for nature…"  │                                      │
│  │  Sources: [urls]                  │                                      │
│  └───────────────────────────────────┘                                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       PIPELINE  (pipeline.py)                               │
│                                                                              │
│  ┌────────────────────┐     ┌─────────────────────┐                          │
│  │ Step 1             │     │ Step 2              │                          │
│  │ Extract Preferences│     │ Embed Query         │                          │
│  │      (LLM Call)    │     │ (SentenceTransformer│                          │
│  │                    │     │  all-MiniLM-L6-v2)  │                          │
│  │  ┌──────────────┐  │     │  ┌───────────────┐  │                          │
│  │  │ {city,budget,│  │     │  │ 384-dim vector│  │                          │
│  │  │  interests}  │  │     │  │ (L2-normed)   │  │                          │
│  │  └──────┬───────┘  │     │  └──────┬────────┘  │                          │
│  └─────────┼──────────┘     └─────────┼───────────┘                          │
│            │                          │                                      │
│            │         ┌────────────────▼──────────────────┐                    │
│            │         │ Step 3: Semantic Search           │                    │
│            │         │ FAISS IndexFlatIP → top 10 chunks │                    │
│            │         └────────────────┬──────────────────┘                    │
│            │                          │                                      │
│            │    ┌─────────────────────▼──────────────────┐                    │
│            └───►│ Step 4: Metadata Filtering             │                    │
│                 │ Hard filter: city + budget + interests  │                    │
│                 └─────────────────────┬──────────────────┘                    │
│                                       │                                      │
│                 ┌─────────────────────▼──────────────────┐                    │
│            ┌───►│ Step 5: Score Re-ranking               │                    │
│            │    │ 0.6×semantic + 0.4×preference_bonus    │                    │
│            │    │ → top 5 chunks                         │                    │
│            │    └─────────────────────┬──────────────────┘                    │
│            │                          │                                      │
│            │    ┌─────────────────────▼──────────────────┐                    │
│            │    │ Step 6: Context Judge (LLM Call)       │                    │
│            │    │ "context_good" or "context_insufficient"│                    │
│            │    └──────┬──────────────────────┬──────────┘                    │
│            │           │                      │                              │
│            │     context_good          context_insufficient                   │
│            │           │                      │                              │
│            │           │              ┌───────▼───────────┐                   │
│            │           │              │ Relax filters     │                   │
│            └───────────┼──────────────│ (drop all filters)│                   │
│                        │              │ retry judge once  │                   │
│                        │              └───────┬───────────┘                   │
│                        │                      │                              │
│                        │              still insufficient → REFUSE             │
│                        │                                                     │
│                 ┌──────▼─────────────────────────────┐                        │
│                 │ Step 7: Generate Answer (LLM Call) │                        │
│                 │ Grounded response + Sources list   │                        │
│                 └────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                   INGESTION (runs once on first start)                       │
│                                                                              │
│   Travel URLs (config.py)                                                    │
│   ┌────────────────────┐                                                     │
│   │ lonelyplanet.com/… │──┐                                                  │
│   │ timeout.com/…      │──┤   fetch_url()      clean_html()    chunk_text()  │
│   │ ...                │──┼──────────────►HTML──────────────►Text────────►Chunks
│   └────────────────────┘  │                                       │          │
│                           │                                       │          │
│     Metadata per URL:     │               embed_texts()           │          │
│     • city                │         (all-MiniLM-L6-v2)            ▼          │
│     • category            │         ┌─────────────────┐    ┌───────────┐     │
│     • price_level         │         │  384-dim vectors │    │ metadata  │     │
│                           │         │  (L2-normalised) │    │ list      │     │
│                           │         └────────┬────────┘    └─────┬─────┘     │
│                           │                  │                   │           │
│                           │                  ▼                   ▼           │
│                           │         ┌─────────────────────────────────┐      │
│                           │         │     data/faiss_index/           │      │
│                           │         │     ├── index.faiss  (vectors) │      │
│                           │         │     └── metadata.json (chunks) │      │
│                           │         └─────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────┐
                        │    EXTERNAL SERVICES     │
                        │                          │
                        │  Groq API (3 LLM calls)  │
                        │  ├─ extract_preferences  │
                        │  ├─ judge_context        │
                        │  └─ generate_answer      │
                        │                          │
                        │  Model: llama-4-scout    │
                        │  -17b-16e-instruct       │
                        └─────────────────────────┘
```

## Component Summary

| File | Role |
|---|---|
| `config.py` | Constants, URL list, model names |
| `embedder.py` | SentenceTransformer wrapper (embed + L2-norm) |
| `ingest.py` | Fetch → Clean → Chunk → FAISS index build |
| `llm.py` | 3 Groq LLM calls (preferences, judge, answer) |
| `retrieval.py` | Semantic search, metadata filter, composite re-rank |
| `pipeline.py` | Orchestrates Steps 1–7 |
| `app.py` | Streamlit UI + cached resource loading |
