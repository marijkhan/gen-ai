"""
Week 3 Day 3 (DAY 13) — Knowledge Graphs
Builds a NetworkX knowledge graph from cat facts and compares KG+RAG vs RAG-only.
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import networkx as nx

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
CAT_FACTS_PATH = SCRIPT_DIR / "cat-facts.txt"
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
    col = chroma.get_or_create_collection("cat-facts-kg", embedding_function=ef)
    if col.count() == 0:
        batch = 50
        for i in range(0, len(facts), batch):
            chunk = facts[i:i + batch]
            col.add(ids=[f"fact-{i+j}" for j in range(len(chunk))], documents=chunk)
    return col


def build_knowledge_graph(facts):
    """Build a knowledge graph from cat facts using LLM-assisted entity/relation extraction."""
    G = nx.DiGraph()

    # Extract entities and relations using the LLM
    # Process facts in batches to reduce API calls
    batch_size = 15
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        facts_text = "\n".join(f"{j+1}. {f}" for j, f in enumerate(batch))

        prompt = f"""Extract entity-relation-entity triples from these cat facts.
Return a JSON array of triples: [{{"subject": "...", "relation": "...", "object": "..."}}]

Use these relation types: has_trait, weighs, lives_for, can_do, has_speed, has_body_part, is_breed, related_to, has_behavior, has_sense

Facts:
{facts_text}

Return ONLY the JSON array, no other text."""

        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            text = match.group(1).strip() if match else raw
            triples = json.loads(text)

            for t in triples:
                if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                    subj = t["subject"].lower().strip()
                    obj = t["object"].lower().strip()
                    rel = t["relation"].lower().strip()
                    G.add_node(subj, type="entity")
                    G.add_node(obj, type="entity")
                    G.add_edge(subj, obj, relation=rel, source_fact=batch[0][:100] if batch else "")
        except Exception as e:
            print(f"  Warning: KG extraction failed for batch {i}: {e}")
            continue

    return G


def query_knowledge_graph(G, query):
    """Query the knowledge graph for relevant triples based on keyword matching."""
    query_terms = set(query.lower().split())
    relevant_triples = []

    for u, v, data in G.edges(data=True):
        node_terms = set(u.split()) | set(v.split())
        overlap = query_terms & node_terms
        if overlap:
            relevant_triples.append({
                "subject": u,
                "relation": data.get("relation", "related_to"),
                "object": v,
            })

    # Also check for partial matches
    if len(relevant_triples) < 3:
        for u, v, data in G.edges(data=True):
            for qt in query_terms:
                if len(qt) > 3 and (qt in u or qt in v):
                    triple = {
                        "subject": u,
                        "relation": data.get("relation", "related_to"),
                        "object": v,
                    }
                    if triple not in relevant_triples:
                        relevant_triples.append(triple)

    return relevant_triples[:10]


def generate_answer(query, context, kg_context=""):
    """Generate answer using context and optional KG triples."""
    full_context = f"Retrieved facts:\n{context}"
    if kg_context:
        full_context += f"\n\nKnowledge graph triples:\n{kg_context}"

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": f"Answer the question using ONLY the context below. If knowledge graph triples are provided, use them to enrich your answer.\n\n{full_context}"},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def check_correctness(answer, ground_truth):
    return ground_truth.lower() in answer.lower()


def main():
    print("=" * 70)
    print("Week 3 Day 3 — Knowledge Graphs")
    print("=" * 70)

    facts = load_facts()
    print(f"Loaded {len(facts)} cat facts.")

    collection = build_chroma_collection(facts)

    print("\nBuilding knowledge graph from cat facts (this may take a moment)...")
    G = build_knowledge_graph(facts)
    print(f"Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results = {
        "kg_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "sample_triples": [],
        },
        "comparison_table": {
            "headers": ["Question", "RAG-Only Correct", "KG+RAG Correct", "KG Triples Used"],
            "rows": [],
        },
        "kg_vs_rag_analysis": {
            "purpose": {
                "rag": "Retrieve semantically similar text chunks for contextual answers",
                "kg": "Capture structured relationships between entities for precise relational queries",
            },
            "ease_of_integration": {
                "rag": "Easy — embed documents, query by similarity",
                "kg": "Moderate — requires entity/relation extraction pipeline, schema design",
            },
            "strengths": {
                "rag": "Handles open-ended questions well, no schema needed, scales with data",
                "kg": "Precise relational queries, multi-hop reasoning, structured data representation",
            },
            "limitations": {
                "rag": "No explicit relationships, may miss connections between distant facts",
                "kg": "Brittle extraction, schema-dependent, struggles with free-form text",
            },
        },
        "when_kg_makes_sense": (
            "KGs add value when: (1) queries involve relationships between entities, "
            "(2) multi-hop reasoning is needed, (3) data has clear structure. "
            "KGs are overkill when: (1) simple factual recall suffices, "
            "(2) data is unstructured prose, (3) the extraction pipeline cost exceeds the benefit."
        ),
        "kg_enhanced_vs_plain_rag_examples": [],
        "detailed_results": [],
    }

    # Sample triples for report
    sample_edges = list(G.edges(data=True))[:10]
    results["kg_stats"]["sample_triples"] = [
        {"subject": u, "relation": d.get("relation", ""), "object": v}
        for u, v, d in sample_edges
    ]

    rag_correct_count, kg_correct_count = 0, 0

    for i, (q, gt) in enumerate(zip(QUESTIONS, GROUND_TRUTH)):
        print(f"\n{'─' * 60}")
        print(f"Q{i+1}: {q}")

        # RAG-only
        retrieval = collection.query(query_texts=[q], n_results=TOP_N)
        rag_context = "\n".join(f"- {d}" for d in retrieval["documents"][0])
        rag_answer = generate_answer(q, rag_context)
        rag_correct = check_correctness(rag_answer, gt)
        rag_correct_count += int(rag_correct)

        # KG + RAG
        kg_triples = query_knowledge_graph(G, q)
        kg_context = "\n".join(f"- {t['subject']} --[{t['relation']}]--> {t['object']}" for t in kg_triples)
        kg_answer = generate_answer(q, rag_context, kg_context)
        kg_correct = check_correctness(kg_answer, gt)
        kg_correct_count += int(kg_correct)

        row = [q, str(rag_correct), str(kg_correct), str(len(kg_triples))]
        results["comparison_table"]["rows"].append(row)

        results["kg_enhanced_vs_plain_rag_examples"].append({
            "question": q,
            "rag_only_answer": rag_answer[:200],
            "kg_rag_answer": kg_answer[:200],
            "kg_triples_used": kg_triples[:3],
            "rag_correct": rag_correct,
            "kg_correct": kg_correct,
            "kg_helped": kg_correct and not rag_correct,
        })

        results["detailed_results"].append({
            "question": q,
            "ground_truth": gt,
            "rag_only": {"answer": rag_answer, "correct": rag_correct},
            "kg_rag": {"answer": kg_answer, "correct": kg_correct, "triples": kg_triples[:5]},
        })

        print(f"  RAG-only correct: {rag_correct} | KG+RAG correct: {kg_correct} | KG triples: {len(kg_triples)}")

    # Print comparison table
    n = len(QUESTIONS)
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE: RAG-Only vs KG+RAG")
    print(f"{'=' * 70}")
    print(f"{'Question':<45} {'RAG':<10} {'KG+RAG':<10} {'KG Triples':<10}")
    print(f"{'─' * 75}")
    for row in results["comparison_table"]["rows"]:
        print(f"{row[0]:<45} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
    print(f"{'─' * 75}")
    print(f"{'TOTAL CORRECT':<45} {rag_correct_count}/{n:<8} {kg_correct_count}/{n:<8}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS: When KGs Make Sense")
    print(f"{'=' * 70}")
    print(f"{'Dimension':<25} {'RAG':<35} {'Knowledge Graph':<35}")
    print(f"{'─' * 95}")
    for dim in ["purpose", "ease_of_integration", "strengths", "limitations"]:
        rag_val = results["kg_vs_rag_analysis"][dim]["rag"][:33]
        kg_val = results["kg_vs_rag_analysis"][dim]["kg"][:33]
        print(f"{dim:<25} {rag_val:<35} {kg_val:<35}")

    output_path = SCRIPT_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
