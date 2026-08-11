# Lab 7 — Multi-Agent Handoff — Joke Generator to Translator

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 — implement agent-to-agent delegation with sub_agents.  
**Tools:** google-adk, sub_agents, Gemini 2.0 Flash

## Goal

Build a three-level agent hierarchy where a root agent hands off to a joke generator, which in turn hands off to a translator. This is the core ADK delegation pattern.

## What you'll build

lab07 — a root agent that produces an English joke and its Chinese translation through two automatic handoffs.

## Setup

Run every command from the `labs/` folder (the parent of this one), so that
the shared virtual environment and the single `.env` are picked up.

```bash
cd ..            # into labs/ if you are inside this lab folder
uv sync          # once per machine — installs google-adk and all deps
cp .env.example .env   # once — then paste your GOOGLE_API_KEY into .env
```

Then run this lab with either:

```bash
uv run adk run lab07     # terminal chat
uv run adk web              # browser IDE, then pick lab07 at http://localhost:8000
```

## Step-by-step

1. Read the three agent definitions and the sub_agents chain

   ```bash
   cat lab07/agent.py
   ```

2. Note that the description field is what the parent reads to decide on a handoff
3. Run the agent in the browser IDE

   ```bash
   uv run adk web
   ```

4. Select lab07 and request a joke

   ```bash
   Tell me a joke
   ```

5. In the Events tab, find the transfer_to_agent call into joke_generator
6. Find the second transfer into translator and confirm the Chinese output
7. Weaken the translator's description to one vague word and re-run
8. Observe the handoff becoming unreliable, then restore the description

## Test it

One 'Tell me a joke' request yields an English joke followed by a Chinese translation, and the Events tab shows two transfer_to_agent calls.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.2 — © 2026 Tertiary Infotech Academy Pte Ltd*
