import numpy as np
import faiss


def semantic_search(
    query_emb: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Search the FAISS index and return top_k chunks with scores.
    query_emb: shape (1, dim), L2-normalised.
    Returns list of chunk dicts with added 'score' field.
    """
    scores, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def apply_metadata_filters(
    candidates: list[dict],
    city: str | None,
    budget: str | None,
    interests: list[str],
) -> list[dict]:
    """
    Hard-filter candidates by city, price_level, and category.
    Falls back to returning the input list unchanged if no candidates pass.
    """
    filtered = list(candidates)

    if city:
        city_lower = city.lower()
        filtered = [c for c in filtered if c.get("city", "").lower() == city_lower]

    if budget and filtered:
        filtered = [c for c in filtered if c.get("price_level", "") == budget]

    if interests and filtered:
        interest_lower = [i.lower() for i in interests]
        interest_filtered = [
            c for c in filtered
            if any(cat in c.get("category", "").lower() for cat in interest_lower)
        ]
        if interest_filtered:
            filtered = interest_filtered

    return filtered if filtered else candidates


def score_rerank(candidates: list[dict], preferences: dict) -> list[dict]:
    """
    Composite score: 0.6 * semantic_score + 0.4 * preference_match_bonus.
    Bonus breakdown:
      +0.3 if city matches
      +0.2 if price_level matches budget
      +0.1 per interest matched (category contains interest keyword)
    """
    city = (preferences.get("city") or "").lower()
    budget = (preferences.get("budget") or "").lower()
    interests = [i.lower() for i in preferences.get("interests", [])]

    scored = []
    for chunk in candidates:
        semantic = chunk.get("score", 0.0)
        bonus = 0.0

        if city and chunk.get("city", "").lower() == city:
            bonus += 0.3

        if budget and chunk.get("price_level", "").lower() == budget:
            bonus += 0.2

        chunk_category = chunk.get("category", "").lower()
        for interest in interests:
            if interest in chunk_category:
                bonus += 0.1

        composite = 0.6 * semantic + 0.4 * bonus
        chunk = dict(chunk)
        chunk["composite_score"] = composite
        scored.append(chunk)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored
