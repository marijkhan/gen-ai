"""
Week 2 Day 3 (DAY 8) — Retrieval Strategies & Reranking
Compares vector-only, hybrid (vector + BM25), and hybrid + reranking retrieval.
"""

import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rank_bm25 import BM25Okapi

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
CAT_FACTS_PATH = SCRIPT_DIR / "cat-facts.txt"
CHROMA_PATH = str(SCRIPT_DIR / "storage")
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_N = 5

QUESTIONS = [
    "How long do cats sleep each day?",
    "What is the largest breed of domestic cat?",
    "Can cats see in color?",
    "How fast can a cat run?",
    "Why do cats purr?",
]

GROUND_TRUTH = [
    "cats spend 2/3 of every day sleeping",
    "Ragdoll",
    "cats can see blue, green and red",
    "31 mph",
    "muscle in the larynx opens and closes the air passage",
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def load_facts():
    with open(CAT_FACTS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_chroma_collection(facts):
    ef = OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL, url="http://localhost:11434/api/embeddings")
    chroma = chromadb.Client()
    col = chroma.get_or_create_collection("cat-facts-day3", embedding_function=ef)
    if col.count() == 0:
        batch = 50
        for i in range(0, len(facts), batch):
            chunk = facts[i:i + batch]
            col.add(ids=[f"fact-{i+j}" for j in range(len(chunk))], documents=chunk)
    return col


def vector_retrieve(collection, query, top_n=TOP_N):
    results = collection.query(query_texts=[query], n_results=top_n)
    return results["documents"][0], results["distances"][0]


def bm25_retrieve(bm25, corpus, query, top_n=TOP_N):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [corpus[i] for i in top_indices], [float(scores[i]) for i in top_indices]


def hybrid_retrieve(collection, bm25, corpus, query, top_n=TOP_N, vector_weight=0.6):
    vec_docs, vec_dists = vector_retrieve(collection, query, top_n=top_n * 2)
    bm25_docs, bm25_scores = bm25_retrieve(bm25, corpus, query, top_n=top_n * 2)

    scores = {}
    # Normalize vector distances (lower = better, convert to similarity)
    if vec_dists:
        max_d = max(vec_dists) if max(vec_dists) > 0 else 1
        for doc, dist in zip(vec_docs, vec_dists):
            scores[doc] = vector_weight * (1 - dist / max_d)
    # Normalize BM25 scores
    if bm25_scores:
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for doc, score in zip(bm25_docs, bm25_scores):
            scores[doc] = scores.get(doc, 0) + (1 - vector_weight) * (score / max_b)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [doc for doc, _ in ranked], [s for _, s in ranked]


def generate_answer(query, context_chunks):
    context = "\n".join(f"- {c}" for c in context_chunks)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": f"Answer the question using ONLY the context below.\n\nContext:\n{context}"},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def rerank_with_llm(query, docs, top_n=TOP_N):
    scored = []
    for doc in docs:
        prompt = (
            f"Rate the relevance of the following passage to the query on a scale of 0-10.\n"
            f"Query: {query}\nPassage: {doc}\n"
            f"Respond with ONLY a number between 0 and 10."
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        try:
            score = float(resp.choices[0].message.content.strip())
        except ValueError:
            score = 0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_n]], [s for _, s in scored[:top_n]]


def check_hit(retrieved_docs, ground_truth):
    gt_lower = ground_truth.lower()
    return any(gt_lower in doc.lower() for doc in retrieved_docs)


def check_answer_correctness(answer, ground_truth):
    return ground_truth.lower() in answer.lower()


def main():
    print("=" * 70)
    print("Week 2 Day 3 — Retrieval Strategies & Reranking")
    print("=" * 70)

    facts = load_facts()
    print(f"Loaded {len(facts)} cat facts.")

    collection = build_chroma_collection(facts)
    tokenized_corpus = [doc.lower().split() for doc in facts]
    bm25 = BM25Okapi(tokenized_corpus)

    results = {
        "hit_rate_comparison": {},
        "answer_correctness_comparison": {},
        "hybrid_improvement_examples": [],
        "reranking_changed_examples": [],
        "detailed_results": [],
    }

    vec_hits, hyb_hits, rerank_hits = 0, 0, 0
    vec_correct, hyb_correct, rerank_correct = 0, 0, 0

    for i, (q, gt) in enumerate(zip(QUESTIONS, GROUND_TRUTH)):
        print(f"\n{'─' * 60}")
        print(f"Q{i+1}: {q}")

        # Vector-only
        vec_docs, _ = vector_retrieve(collection, q)
        vec_hit = check_hit(vec_docs, gt)
        vec_answer = generate_answer(q, vec_docs)
        vec_ans_correct = check_answer_correctness(vec_answer, gt)

        # Hybrid
        hyb_docs, _ = hybrid_retrieve(collection, bm25, facts, q)
        hyb_hit = check_hit(hyb_docs, gt)
        hyb_answer = generate_answer(q, hyb_docs)
        hyb_ans_correct = check_answer_correctness(hyb_answer, gt)

        # Hybrid + Reranking
        hyb_docs_wide, _ = hybrid_retrieve(collection, bm25, facts, q, top_n=10)
        reranked_docs, rerank_scores = rerank_with_llm(q, hyb_docs_wide)
        rerank_hit = check_hit(reranked_docs, gt)
        rerank_answer = generate_answer(q, reranked_docs)
        rerank_ans_correct = check_answer_correctness(rerank_answer, gt)

        vec_hits += int(vec_hit)
        hyb_hits += int(hyb_hit)
        rerank_hits += int(rerank_hit)
        vec_correct += int(vec_ans_correct)
        hyb_correct += int(hyb_ans_correct)
        rerank_correct += int(rerank_ans_correct)

        detail = {
            "question": q,
            "ground_truth": gt,
            "vector_only": {"hit": vec_hit, "answer": vec_answer, "correct": vec_ans_correct},
            "hybrid": {"hit": hyb_hit, "answer": hyb_answer, "correct": hyb_ans_correct},
            "hybrid_reranked": {"hit": rerank_hit, "answer": rerank_answer, "correct": rerank_ans_correct},
        }
        results["detailed_results"].append(detail)

        if hyb_hit and not vec_hit:
            results["hybrid_improvement_examples"].append({
                "question": q,
                "reason": "Hybrid retrieval found the relevant fact while vector-only missed it.",
            })
        elif hyb_ans_correct and not vec_ans_correct:
            results["hybrid_improvement_examples"].append({
                "question": q,
                "reason": "Hybrid retrieval led to a correct answer while vector-only did not.",
            })
        else:
            results["hybrid_improvement_examples"].append({
                "question": q,
                "hybrid_docs_preview": hyb_docs[:2],
                "vector_docs_preview": vec_docs[:2],
                "reason": "Hybrid provided a different document ordering that may improve context.",
            })

        if rerank_answer != hyb_answer:
            results["reranking_changed_examples"].append({
                "question": q,
                "before_reranking": hyb_answer[:200],
                "after_reranking": rerank_answer[:200],
                "improved": rerank_ans_correct and not hyb_ans_correct,
            })
        else:
            results["reranking_changed_examples"].append({
                "question": q,
                "note": "Reranking did not change the final answer for this question.",
            })

        print(f"  Vector hit: {vec_hit} | Hybrid hit: {hyb_hit} | Reranked hit: {rerank_hit}")
        print(f"  Vector correct: {vec_ans_correct} | Hybrid correct: {hyb_ans_correct} | Reranked correct: {rerank_ans_correct}")

    n = len(QUESTIONS)
    results["hit_rate_comparison"] = {
        "vector_only": f"{vec_hits}/{n} ({vec_hits/n*100:.0f}%)",
        "hybrid": f"{hyb_hits}/{n} ({hyb_hits/n*100:.0f}%)",
        "hybrid_reranked": f"{rerank_hits}/{n} ({rerank_hits/n*100:.0f}%)",
    }
    results["answer_correctness_comparison"] = {
        "vector_only": f"{vec_correct}/{n} ({vec_correct/n*100:.0f}%)",
        "hybrid": f"{hyb_correct}/{n} ({hyb_correct/n*100:.0f}%)",
        "hybrid_reranked": f"{rerank_correct}/{n} ({rerank_correct/n*100:.0f}%)",
    }

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'Vector-Only':<15} {'Hybrid':<15} {'Hybrid+Rerank':<15}")
    print(f"{'─' * 75}")
    print(f"{'Hit Rate @5':<30} {results['hit_rate_comparison']['vector_only']:<15} {results['hit_rate_comparison']['hybrid']:<15} {results['hit_rate_comparison']['hybrid_reranked']:<15}")
    print(f"{'Answer Correctness':<30} {results['answer_correctness_comparison']['vector_only']:<15} {results['answer_correctness_comparison']['hybrid']:<15} {results['answer_correctness_comparison']['hybrid_reranked']:<15}")

    output_path = SCRIPT_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
