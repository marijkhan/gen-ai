# Naive Stuffing vs Summarize-Then-Answer: Comparative Analysis

## Overview

This report compares two RAG strategies for answering questions from a large API reference document (~48,530 characters, ~14k tokens):

1. **Naive Stuffing**: Inject the entire document into context
2. **Summarize-Then-Answer**: First summarize the document, then answer from the summary

## Results by Question Type

### 1. Specific Lookup Questions
*Example: "Does WiFi successful connection event return the BSSID?"*

| Strategy | Result | Confidence | Tokens Used |
|----------|--------|------------|-------------|
| Naive Stuffing | Correct | High | ~14k |
| Summarize-Then-Answer | Uncertain | Low | ~1k |

**Winner: Naive Stuffing**

The summarization process loses granular details. The model hedged with phrases like "it is likely" and "to confirm, review the actual API response."

---

### 2. Counting/Enumeration Questions
*Example: "How many different types of events are there?"*

| Strategy | Result | Tokens Used |
|----------|--------|-------------|
| Naive Stuffing | Incorrect (said 37, listed 28) | ~14k |
| Summarize-Then-Answer | Correct (28) | ~1k |

**Winner: Summarize-Then-Answer**

Naive stuffing hallucinated a count (37) that didn't match its own enumeration (28). The summarization step forced the model to consolidate and count correctly upfront.

---

### 3. Aggregation Questions
*Example: "How many events return the STA MAC address?"*

| Strategy | Result | Tokens Used |
|----------|--------|-------------|
| Naive Stuffing | Incorrect (said 28, correct is 25) | ~14k |
| Summarize-Then-Answer | Not tested | - |

**Winner: Neither (both likely to struggle)**

Aggregation across a large context is error-prone. The model listed all 28 events instead of filtering for those containing the specific field.

---

## Key Findings

| Question Type | Best Strategy | Why |
|---------------|---------------|-----|
| Specific lookups | Naive Stuffing | Preserves exact details |
| Counting totals | Summarize-Then-Answer | Forces upfront consolidation |
| Cross-document aggregation | Neither reliable | Requires structured extraction |

## Interesting Observation

When challenged with incorrect information (run 4), the model resisted manipulation and answered correctly with confidence. This suggests the model has some robustness against adversarial prompts when the source document is in context.

---

## Recommendations for Improvement

### For Naive Stuffing
1. **Add explicit counting prompts**: Prepend instructions like "Count carefully before stating a number"
2. **Use structured output**: Request JSON with enumerated lists to force systematic traversal
3. **Chain-of-thought for aggregation**: Ask model to list matching items first, then count

### For Summarize-Then-Answer
1. **Use hierarchical summaries**: Keep a detailed summary + a high-level overview
2. **Preserve key fields in summary**: Explicitly list all event names and their core fields
3. **Hybrid approach**: For specific lookups, fall back to the original document

### General Improvements
1. **Chunked retrieval**: Split document into sections, retrieve only relevant chunks
2. **Field extraction preprocessing**: For structured docs (APIs), extract into a searchable format first
3. **Multi-pass verification**: For counting questions, run the query twice and compare results

---

## Token Efficiency

| Strategy | Tokens per Query | Best For |
|----------|------------------|----------|
| Naive Stuffing | ~14,000 | Precision-critical lookups |
| Summarize-Then-Answer | ~1,000 | High-level questions, cost-sensitive applications |

Summarize-then-answer uses **~14x fewer tokens** per query after initial summarization, making it significantly more cost-effective for repeated queries against the same document.

---

## Conclusion

Neither strategy dominates across all question types. The optimal approach depends on:
- **Query type**: Specific vs aggregate
- **Accuracy requirements**: Can you tolerate hedged answers?
- **Cost constraints**: Token budget matters

A hybrid system that routes queries to the appropriate strategy based on question classification would likely outperform either approach alone.
