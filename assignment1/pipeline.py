import logging

import faiss

from config import TOP_K_RERANK, TOP_K_RETRIEVAL
from embedder import embed_query
from llm import extract_preferences, generate_answer, judge_context
from retrieval import apply_metadata_filters, score_rerank, semantic_search

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I'm sorry, but I don't have enough information in my knowledge base to give "
    "reliable recommendations for your query. My knowledge currently covers travel "
    "destinations in Berlin, Paris, Barcelona, and Tokyo. Please try a query about "
    "one of those cities, or broaden your search criteria."
)


def run_pipeline(query: str, index: faiss.Index, metadata: list[dict]) -> dict:
    """
    Full RAG pipeline. Returns a result dict with keys:
      answer, preferences, context_verdict, chunks, filters_relaxed
    """
    result = {
        "answer": "",
        "preferences": {},
        "context_verdict": "",
        "chunks": [],
        "filters_relaxed": False,
    }

    # Step 1: Extract preferences
    logger.info("Step 1: Extracting preferences...")
    preferences = extract_preferences(query)
    result["preferences"] = preferences
    logger.info(f"  Preferences: {preferences}")

    # Step 2: Embed query
    logger.info("Step 2: Embedding query...")
    query_emb = embed_query(query)

    # Step 3: Semantic search
    logger.info("Step 3: Semantic search...")
    candidates = semantic_search(query_emb, index, metadata, top_k=TOP_K_RETRIEVAL)

    # Step 4: Metadata filtering
    logger.info("Step 4: Applying metadata filters...")
    filtered = apply_metadata_filters(
        candidates,
        city=preferences.get("city"),
        budget=preferences.get("budget"),
        interests=preferences.get("interests", []),
    )

    # Step 5: Re-rank, take top 5
    logger.info("Step 5: Re-ranking...")
    if len(filtered) < 3:
        logger.info("  Fewer than 3 filtered results; falling back to unfiltered candidates")
        reranked = score_rerank(candidates, preferences)[:TOP_K_RERANK]
    else:
        reranked = score_rerank(filtered, preferences)[:TOP_K_RERANK]

    result["chunks"] = reranked

    # Step 6: Judge context quality
    logger.info("Step 6: Judging context...")
    chunk_texts = [c["text"] for c in reranked]
    verdict = judge_context(query, chunk_texts)
    result["context_verdict"] = verdict
    logger.info(f"  Verdict: {verdict}")

    if verdict == "context_insufficient":
        # Relax filters to city-only and retry
        logger.info("  Context insufficient. Relaxing filters to city-only...")
        city = preferences.get("city")
        relaxed = [c for c in candidates if not city or c.get("city", "").lower() == (city or "").lower()]
        reranked_relaxed = score_rerank(relaxed if relaxed else candidates, preferences)[:TOP_K_RERANK]
        chunk_texts_relaxed = [c["text"] for c in reranked_relaxed]

        verdict2 = judge_context(query, chunk_texts_relaxed)
        result["context_verdict"] = verdict2
        logger.info(f"  Second verdict: {verdict2}")

        if verdict2 == "context_insufficient":
            result["answer"] = REFUSAL_MESSAGE
            return result

        result["chunks"] = reranked_relaxed
        result["filters_relaxed"] = True
        reranked = reranked_relaxed

    # Step 7: Generate answer
    logger.info("Step 7: Generating answer...")
    answer = generate_answer(query, preferences, reranked)
    result["answer"] = answer

    return result
