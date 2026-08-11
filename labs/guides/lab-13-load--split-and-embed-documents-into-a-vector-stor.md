# Lab 13 — Load, Split and Embed Documents into a Vector Store

**Topic 03:** Build Agentic AI RAG in Gemini ADK  
**Learning outcome:** LO4 — build the ingestion half of a RAG pipeline and inspect the embeddings.  
**Agent folder:** `labs/agent_rag`  
**Tools:** ChromaDB, pypdf, google-adk, Gemini embeddings

## Goal

Take two product PDFs, extract their text, split it into retrievable chunks with page metadata, embed them and persist the vectors in a local Chroma database.

## What you'll build

A persistent Chroma collection holding the chunked and embedded air-fryer product and warranty manuals.

## Step-by-step

1. Inspect the two source PDFs the agent will be grounded in

   ```bash
   ls agent_rag/*.pdf
   ```

2. Read the extraction function and note the page number kept as metadata

   ```bash
   cat agent_rag/agent.py
   ```

3. Identify the chunking rule and the minimum chunk length filter
4. Run the agent once to trigger ingestion into ChromaDB

   ```bash
   uv run adk run agent_rag
   ```

5. Confirm the persistent vector store was created on disk

   ```bash
   ls agent_rag/chroma_db
   ```

6. Explain why chunks shorter than 50 characters are discarded
7. Change the chunk rule to split on single newlines, delete chroma_db, and re-ingest

   ```bash
   rm -rf agent_rag/chroma_db
   ```

8. Compare the resulting chunk count and note the effect on retrieval precision

## Test it

agent_rag/chroma_db exists and contains a populated air_fryer_docs collection, and you can state how the chunking rule changed the number of stored chunks.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
