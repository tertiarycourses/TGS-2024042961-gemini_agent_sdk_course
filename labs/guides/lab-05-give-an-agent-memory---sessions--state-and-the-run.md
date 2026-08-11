# Lab 5 — Give an Agent Memory — Sessions, State and the Runner

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 — implement session management so an agent remembers earlier turns.  
**Agent folder:** `labs/agent_session`  
**Tools:** google-adk, Runner, InMemorySessionService, google.genai types

## Goal

Drive an agent programmatically with a Runner and an InMemorySessionService, so the conversation persists across turns instead of restarting on every message.

## What you'll build

agent_session — a multi-turn agent that answers follow-up questions using earlier context.

## Step-by-step

1. Read how the session service, session and Runner are wired together

   ```bash
   cat agent_session/agent.py
   ```

2. Note the APP_NAME, USER_ID and SESSION_ID that identify one conversation
3. Run the session script

   ```bash
   uv run python agent_session/agent.py
   ```

4. Ask an initial question that establishes context

   ```bash
   What is the weather in Tokyo?
   ```

5. Ask a follow-up that only works if the agent remembers

   ```bash
   And what about Osaka?
   ```

6. Comment out the session creation and re-run to observe the failure
7. Restore the session code and confirm continuity is back

## Test it

The follow-up 'And what about Osaka?' is understood as a weather question without repeating the word weather — and stops working when the session is removed.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
