# Lab 17 — Ship the Agent as a Web App with Streamlit

**Topic 04:** Build an Agentic AI App with Gemini Agent ADK and Streamlit  
**Learning outcome:** LO3 — deploy a multi-agent system behind a usable chat interface.  
**Agent folder:** `labs/transport_agent_streamlit`  
**Tools:** Streamlit, google-adk Runner, InMemorySessionService, asyncio

## Goal

Wrap the sequential transport agent in a Streamlit chat UI, wiring the async ADK Runner and persisting the session across Streamlit re-runs so the conversation survives.

## What you'll build

transport_agent_streamlit — a browser chat application backed by the multi-agent transport workflow.

## Step-by-step

1. Read how the ADK session service is stored in st.session_state

   ```bash
   cat transport_agent_streamlit/app.py
   ```

2. Note that the async Runner is driven through asyncio.run inside the handler
3. Launch the Streamlit application

   ```bash
   uv run streamlit run transport_agent_streamlit/app.py
   ```

4. Open the app in the browser and submit a journey

   ```bash
   From Tampines to Orchard Road
   ```

5. Send a follow-up question and confirm the conversation history is retained

   ```bash
   What if I cycle instead?
   ```

6. Refresh the browser and observe what happens to the session
7. Change the page title and icon in st.set_page_config and reload
8. Add a sidebar Clear chat button that resets st.session_state.messages

## Test it

The Streamlit app answers a journey query, retains history across follow-up turns, and your Clear chat button empties the conversation without restarting the server.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
