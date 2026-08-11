# Lab 4 — Swap the Model — Running an ADK Agent on a Non-Gemini LLM

**Topic 01:** Overview of Agentic AI in Gemini ADK  
**Learning outcome:** LO2 / LO4 — compare models and assess the trade-offs for an engineering process.  
**Agent folder:** `labs/agent_model`  
**Tools:** google-adk, LiteLlm, Gemini 2.0 Flash, OpenAI GPT-4.1-mini

## Goal

Use the LiteLlm wrapper to point the same agent at an OpenAI model, then compare it with Gemini on the same prompts to judge quality, latency and cost.

## What you'll build

agent_model — one agent definition that runs on either Gemini or an OpenAI model by changing a single line.

## Step-by-step

1. Inspect how LiteLlm wraps a non-Google model

   ```bash
   cat agent_model/agent.py
   ```

2. Add your OpenAI key to labs/.env

   ```bash
   echo 'OPENAI_API_KEY=your-openai-key' >> .env
   ```

3. Run the agent on the OpenAI model

   ```bash
   uv run adk run agent_model
   ```

4. Ask a reasoning question and note the answer quality and response time

   ```bash
   Explain in three sentences why an agent needs tools.
   ```

5. Edit agent_model/agent.py and replace the model with a Gemini model string

   ```bash
   model='gemini-2.0-flash'
   ```

6. Re-run the identical prompt on Gemini

   ```bash
   uv run adk run agent_model
   ```

7. Record quality, latency and cost for both in a comparison table

## Test it

The same agent runs unchanged on both providers, and you can state which model you would choose for this workload and justify it on quality, latency and cost.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
