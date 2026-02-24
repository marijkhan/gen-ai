import logging
import os
import sys

# Ensure imports resolve from assignment1/ directory
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st

from config import FAISS_INDEX_DIR
from embedder import get_model
from ingest import build_index, index_exists, load_index, save_index
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Travel RAG Assistant", page_icon="✈", layout="centered")
st.title("✈ Travel RAG Assistant")
st.caption("Ask me for travel recommendations — I'll find grounded suggestions from real sources.")

# ── Resolve index path relative to this file ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, FAISS_INDEX_DIR)


# ── Cached resources (load once per server process) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return get_model()


@st.cache_resource(show_spinner=False)
def load_or_build_index():
    if index_exists(INDEX_PATH):
        return load_index(INDEX_PATH)
    # First-run: fetch and index travel content
    with st.spinner("Fetching and indexing travel data... (this runs once)"):
        index, metadata = build_index()
        save_index(index, metadata, INDEX_PATH)
    return index, metadata


# Eagerly initialise (shows spinner on first run)
load_embedder()
index, metadata = load_or_build_index()

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. cheap food in Berlin, I love street art",
    key="query_input",
)

run_button = st.button("Find Recommendations", type="primary", disabled=not query.strip())

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run_button and query.strip():
    with st.spinner("Searching for recommendations..."):
        result = run_pipeline(query.strip(), index, metadata)

    st.divider()

    # Relaxed-filter warning
    if result.get("filters_relaxed"):
        st.warning(
            "Filters were relaxed (only city was applied) because the initial search "
            "returned insufficient context. Results may be broader than requested."
        )

    # Main answer
    st.subheader("Recommendation")
    if result["answer"]:
        st.markdown(result["answer"])
    else:
        st.info("No answer was generated.")

    # ── Debug panel ───────────────────────────────────────────────────────────
    with st.expander("Debug Panel", expanded=False):
        st.markdown("**Extracted Preferences**")
        st.json(result["preferences"])

        st.markdown(f"**Context Verdict:** `{result['context_verdict']}`")

        chunks = result.get("chunks", [])
        if chunks:
            urls_used = list(dict.fromkeys(c["url"] for c in chunks))
            st.markdown("**URLs Used**")
            for url in urls_used:
                st.markdown(f"- {url}")

            st.markdown("**Top Chunks**")
            for i, chunk in enumerate(chunks, 1):
                with st.container():
                    cols = st.columns([1, 1, 1, 1])
                    cols[0].markdown(f"**#{i}** `{chunk['city']}`")
                    cols[1].markdown(f"`{chunk['category']}`")
                    cols[2].markdown(f"`{chunk['price_level']}`")
                    cols[3].markdown(f"score: `{chunk.get('composite_score', chunk.get('score', 0)):.3f}`")
                    st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
                    st.divider()
