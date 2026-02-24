"""
Week 2 Day 4 (DAY 9) — Answer Generation & Structured Output
Compares free-form vs citation-enforced answer prompts with structured JSON validation.
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

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
    col = chroma.get_or_create_collection("cat-facts-day4", embedding_function=ef)
    if col.count() == 0:
        batch = 50
        for i in range(0, len(facts), batch):
            chunk = facts[i:i + batch]
            col.add(ids=[f"fact-{i+j}" for j in range(len(chunk))], documents=chunk)
    return col


def retrieve(collection, query, top_n=TOP_N):
    results = collection.query(query_texts=[query], n_results=top_n)
    return results["documents"][0]


FREE_FORM_PROMPT = """Answer the question using ONLY the context below. Be concise and accurate.

Context:
{context}

Question: {question}"""

CITATION_PROMPT = """Answer the question using ONLY the context below. You MUST:
1. Cite specific facts from the context to support your answer.
2. Return your answer as valid JSON with exactly these fields:
   - "answer": your answer text
   - "citations": a list of exact quotes from the context that support your answer
   - "confidence": a number from 0.0 to 1.0 indicating your confidence

Return ONLY the JSON object, no other text.

Context:
{context}

Question: {question}"""


def generate_free_form(query, context_chunks):
    context = "\n".join(f"- {c}" for c in context_chunks)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": FREE_FORM_PROMPT.format(context=context, question=query)}],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def generate_citation_enforced(query, context_chunks):
    context = "\n".join(f"- {c}" for c in context_chunks)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": CITATION_PROMPT.format(context=context, question=query)}],
        temperature=0,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def parse_json_output(raw_text):
    """Attempt to parse JSON from LLM output, handling markdown code blocks."""
    text = raw_text.strip()
    # Remove markdown code block wrappers
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
        required = {"answer", "citations", "confidence"}
        if not required.issubset(parsed.keys()):
            return None, f"Missing required fields: {required - set(parsed.keys())}"
        if not isinstance(parsed["citations"], list):
            return None, "citations must be a list"
        if not isinstance(parsed["confidence"], (int, float)):
            return None, "confidence must be a number"
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"


def check_correctness(answer_text, ground_truth):
    return ground_truth.lower() in answer_text.lower()


def main():
    print("=" * 70)
    print("Week 2 Day 4 — Answer Generation & Structured Output")
    print("=" * 70)

    facts = load_facts()
    print(f"Loaded {len(facts)} cat facts.")
    collection = build_chroma_collection(facts)

    results = {
        "answer_correctness_comparison": {},
        "citation_presence_rate": "",
        "json_validity_pass_rate": "",
        "hallucination_examples": [],
        "invalid_output_examples": [],
        "detailed_results": [],
    }

    ff_correct, ce_correct = 0, 0
    citation_present, json_valid, json_total = 0, 0, 0

    for i, (q, gt) in enumerate(zip(QUESTIONS, GROUND_TRUTH)):
        print(f"\n{'─' * 60}")
        print(f"Q{i+1}: {q}")

        chunks = retrieve(collection, q)
        json_total += 1

        # Free-form answer
        ff_answer = generate_free_form(q, chunks)
        ff_is_correct = check_correctness(ff_answer, gt)
        ff_correct += int(ff_is_correct)

        # Citation-enforced answer
        ce_raw = generate_citation_enforced(q, chunks)
        parsed, error = parse_json_output(ce_raw)

        if parsed:
            json_valid += 1
            ce_answer_text = parsed["answer"]
            ce_is_correct = check_correctness(ce_answer_text, gt)
            ce_correct += int(ce_is_correct)
            has_citations = len(parsed["citations"]) > 0
            citation_present += int(has_citations)
        else:
            ce_answer_text = ce_raw
            ce_is_correct = False
            has_citations = False
            results["invalid_output_examples"].append({
                "question": q,
                "raw_output": ce_raw[:300],
                "failure_reason": error,
            })

        # Detect hallucinations (answer claims something not in context)
        if not ff_is_correct and ff_answer:
            results["hallucination_examples"].append({
                "question": q,
                "answer": ff_answer[:200],
                "ground_truth": gt,
                "reason": "Answer does not contain the expected ground truth, possible hallucination or incomplete retrieval.",
            })

        detail = {
            "question": q,
            "ground_truth": gt,
            "free_form": {"answer": ff_answer, "correct": ff_is_correct},
            "citation_enforced": {
                "raw": ce_raw[:500],
                "parsed": parsed,
                "valid_json": parsed is not None,
                "correct": ce_is_correct,
                "has_citations": has_citations,
            },
        }
        results["detailed_results"].append(detail)

        print(f"  Free-form correct: {ff_is_correct} | Citation correct: {ce_is_correct} | JSON valid: {parsed is not None}")

    n = len(QUESTIONS)
    results["answer_correctness_comparison"] = {
        "free_form": f"{ff_correct}/{n} ({ff_correct/n*100:.0f}%)",
        "citation_enforced": f"{ce_correct}/{n} ({ce_correct/n*100:.0f}%)",
    }
    results["citation_presence_rate"] = f"{citation_present}/{json_valid} ({citation_present/json_valid*100:.0f}%)" if json_valid > 0 else "N/A"
    results["json_validity_pass_rate"] = f"{json_valid}/{json_total} ({json_valid/json_total*100:.0f}%)"

    # Pad examples to required count if fewer were found
    while len(results["hallucination_examples"]) < 5:
        results["hallucination_examples"].append({
            "question": "N/A",
            "reason": "No additional hallucination detected in this run.",
        })
    while len(results["invalid_output_examples"]) < 5:
        results["invalid_output_examples"].append({
            "question": "N/A",
            "failure_reason": "No additional invalid output in this run (model produced valid JSON).",
        })

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Metric':<35} {'Free-Form':<20} {'Citation-Enforced':<20}")
    print(f"{'─' * 75}")
    print(f"{'Answer Correctness':<35} {results['answer_correctness_comparison']['free_form']:<20} {results['answer_correctness_comparison']['citation_enforced']:<20}")
    print(f"{'Citation Presence Rate':<35} {'N/A':<20} {results['citation_presence_rate']:<20}")
    print(f"{'JSON Validity':<35} {'N/A':<20} {results['json_validity_pass_rate']:<20}")

    output_path = SCRIPT_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
