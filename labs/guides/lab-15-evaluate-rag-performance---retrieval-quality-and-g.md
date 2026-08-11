# Lab 15 — Evaluate RAG Performance — Retrieval Quality and Groundedness

**Topic 03:** Build Agentic AI RAG in Gemini ADK  
**Learning outcome:** LO4 — evaluate the performance effectiveness of a RAG implementation.  
**Agent folder:** `labs/agent_rag`  
**Tools:** agent_rag, ChromaDB, similarity search vs MMR

## Goal

Build a small evaluation set and score your RAG agent on retrieval hit rate, groundedness, answer relevance and latency, then tune one parameter and measure the improvement.

## What you'll build

A completed RAG evaluation table with a before-and-after comparison for one tuned parameter.

## Step-by-step

1. Write eight test questions: six answerable from the PDFs, two deliberately out of scope
2. For each question record whether the correct chunk was retrieved (retrieval hit)
3. For each answer record whether every claim is supported by the retrieved text (groundedness)
4. Record whether the answer actually addresses the question (relevance) and its response time
5. Compute the hit rate, groundedness rate and mean latency across the eight questions
6. Tune one parameter — chunk size, n_results, or switch similarity search to MMR
7. Re-run the same eight questions and recompute all three metrics
8. State which parameter change you would keep and justify it with your numbers

## Test it

A completed evaluation table with before-and-after figures for retrieval hit rate, groundedness and latency, plus a written, evidence-based tuning recommendation.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
