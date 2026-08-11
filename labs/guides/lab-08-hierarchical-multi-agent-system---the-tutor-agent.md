# Lab 8 — Hierarchical Multi-Agent System — The Tutor Agent

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 — design a coordinator agent routing to specialised sub-agents.  
**Agent folder:** `labs/tutor_agent`  
**Tools:** google-adk, sub_agents, coordinator pattern

## Goal

Build a tutoring system where one root agent routes each question to a maths, physics or history specialist, each with its own instruction and teaching style.

## What you'll build

tutor_agent — a coordinator with three subject specialists that routes by topic.

## Step-by-step

1. Read the three specialist agents and the root coordinator

   ```bash
   cat tutor_agent/agent.py
   ```

2. Compare the three descriptions and note how each states its routing trigger
3. Launch the web IDE and select tutor_agent

   ```bash
   uv run adk web
   ```

4. Ask a maths question and confirm it routes to math_tutor_agent

   ```bash
   Solve 2x + 5 = 17 step by step.
   ```

5. Ask a physics question

   ```bash
   Explain Newton's second law with an example.
   ```

6. Ask a history question

   ```bash
   What caused the fall of the Roman Empire?
   ```

7. Ask an ambiguous cross-subject question and observe how the router resolves it

   ```bash
   How did physics change during the Industrial Revolution?
   ```

8. Add a fourth specialist of your own choosing to sub_agents and test the routing

## Test it

Each subject question is answered by the matching specialist, visible as a transfer_to_agent event, and your new fourth specialist is routed to correctly.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
