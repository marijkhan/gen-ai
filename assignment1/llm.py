import json
import logging
import os

from groq import Groq

from config import GROQ_MODEL

logger = logging.getLogger(__name__)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def extract_preferences(query: str, client: Groq | None = None) -> dict:
    """
    Extract structured preferences from a natural-language travel query.
    Returns: {"city": str | null, "budget": str | null, "interests": [str]}
    """
    client = client or get_client()
    system_prompt = (
        "You are a travel preference extractor. "
        "Given a user query, extract travel preferences as JSON.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"city": <city name in lowercase or null>, '
        '"budget": <"cheap"|"medium"|"expensive" or null>, '
        '"interests": [<list of interest strings>]}\n'
        "Interests can include: food, art, sightseeing, nightlife, shopping, nature, museums, etc.\n"
        "Do not include any explanation, only the JSON object."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        prefs = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse preferences JSON: {raw!r}")
        prefs = {"city": None, "budget": None, "interests": []}

    # Normalise
    prefs.setdefault("city", None)
    prefs.setdefault("budget", None)
    prefs.setdefault("interests", [])
    if isinstance(prefs["interests"], str):
        prefs["interests"] = [prefs["interests"]]

    return prefs


def judge_context(query: str, chunks: list[str], client: Groq | None = None) -> str:
    """
    Assess whether retrieved chunks are sufficient to answer the query.
    Returns "context_good" or "context_insufficient".
    """
    client = client or get_client()
    context_preview = "\n---\n".join(chunks[:5])

    system_prompt = (
        "You are a context quality judge for a travel recommendation system.\n"
        "Given a user query and retrieved context chunks, decide if the context "
        "contains enough relevant information to give a helpful answer.\n"
        'Reply with ONLY one of: "context_good" or "context_insufficient".\n'
        "No explanation, just the label."
    )
    user_message = f"Query: {query}\n\nContext:\n{context_preview}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    verdict = response.choices[0].message.content.strip().lower()
    if "insufficient" in verdict:
        return "context_insufficient"
    return "context_good"


def generate_answer(
    query: str,
    preferences: dict,
    chunks: list[dict],
    client: Groq | None = None,
) -> str:
    """
    Generate a travel recommendation answer grounded in the retrieved chunks.
    Each chunk dict must have keys: text, url, city, category, price_level.
    """
    client = client or get_client()

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['url']} | {chunk['city']} | {chunk['category']} | {chunk['price_level']}]\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    pref_summary = (
        f"City: {preferences.get('city') or 'not specified'}, "
        f"Budget: {preferences.get('budget') or 'not specified'}, "
        f"Interests: {', '.join(preferences.get('interests', [])) or 'not specified'}"
    )

    system_prompt = (
        "You are an expert travel advisor. Using ONLY the provided context, "
        "give a helpful, specific travel recommendation that matches the user's preferences.\n"
        "- Cite sources using [Source N] inline.\n"
        "- At the end, list the source URLs under a 'Sources:' section.\n"
        "- Be concise but informative (aim for 200-350 words).\n"
        "- If the context doesn't cover something, say so honestly rather than inventing details."
    )

    user_message = (
        f"User query: {query}\n"
        f"User preferences: {pref_summary}\n\n"
        f"Context:\n{context_str}"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=700,
    )

    return response.choices[0].message.content.strip()
