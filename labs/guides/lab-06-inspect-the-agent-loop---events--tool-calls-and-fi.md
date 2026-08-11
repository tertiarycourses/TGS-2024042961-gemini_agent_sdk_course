# Lab 6 — Inspect the Agent Loop — Events, Tool Calls and Final Responses

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 / LO4 — analyse the agent execution loop and evaluate its behaviour.  
**Agent folder:** `labs/agent_interact`  
**Tools:** google-adk, Runner, Events API, InMemorySessionService

## Goal

Stream the Event objects the Runner emits and learn to read an agent trace: which event carries the tool call, which carries the tool result, and which is the final response.

## What you'll build

agent_interact — an instrumented agent that prints every event in its reasoning loop.

## Step-by-step

1. Read the event-handling loop and find is_final_response()

   ```bash
   cat agent_interact/agent.py
   ```

2. Run the interaction script

   ```bash
   uv run python agent_interact/agent.py
   ```

3. Send a prompt that requires a tool

   ```bash
   Find the weather in Singapore and summarise recent AI news.
   ```

4. In the printed trace, identify the function_call event
5. Identify the function_response event carrying the tool's return value
6. Identify the final response event and note how many LLM calls one turn actually took
7. Explain why a two-tool question produces more events than a one-tool question

## Test it

You can point to the function_call, the function_response and the final response in the trace, and state the number of LLM round-trips the turn consumed.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
