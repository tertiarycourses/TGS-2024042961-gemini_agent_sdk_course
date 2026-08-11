# Lab 9 — Sequential Workflow Agent — Singapore Transport Route Planner

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 / LO2 — orchestrate a fixed multi-stage pipeline with SequentialAgent.  
**Agent folder:** `labs/transport_agent`  
**Tools:** google-adk, SequentialAgent, LlmAgent, google_search

## Goal

Chain three agents in a fixed order — collect the journey, research cross-country options, then produce a full route report — using SequentialAgent with the google_search tool.

## What you'll build

transport_agent — a sequential pipeline producing a route report by MRT, bus, taxi, cycling and walking.

## Step-by-step

1. Read the three sub-agents and the SequentialAgent that orders them

   ```bash
   cat transport_agent/agent.py
   ```

2. Note that each agent's output is passed forward as the next agent's input
3. Launch the web IDE and select transport_agent

   ```bash
   uv run adk web
   ```

4. Provide a journey when the first agent asks

   ```bash
   From Jurong East to Changi Airport
   ```

5. Wait for all three stages to complete and read the consolidated report
6. Verify the report covers bus, MRT, taxi, cycling, walking and a fastest route
7. Explain why SequentialAgent, not sub_agents handoff, is the right pattern here

## Test it

A single journey request produces one report containing all five transport modes plus a recommended fastest route, with the three stages visible in order in the Events tab.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
