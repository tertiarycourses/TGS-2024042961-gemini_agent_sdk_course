# Lab 3 — Give an Agent Tools — Live Weather and Web Search

**Topic 01:** Overview of Agentic AI in Gemini ADK  
**Learning outcome:** LO1 / LO3 — extend an agent with custom function tools and evaluate tool selection.  
**Agent folder:** `labs/multi_tools_agent`  
**Tools:** google-adk, OpenWeather API, Tavily API, Gemini 2.0 Flash

## Goal

Add two Python function tools to an agent: a live OpenWeather lookup and a Tavily web search. You see how the docstring and type hints become the tool contract the model reads.

## What you'll build

multi_tools_agent — an agent that decides for itself whether a question needs the weather tool, the search tool, both, or neither.

## Step-by-step

1. Register for a free OpenWeather API key at openweathermap.org and a Tavily key at tavily.com
2. Add both keys to labs/.env as OPENWEATHER_API_KEY and TAVILY_API_KEY
3. Read the two tool functions and note the docstring, the typed arguments and the dict return

   ```bash
   cat multi_tools_agent/agent.py
   ```

4. Observe that tools are attached with tools=[get_weather, tavily_search]
5. Run the agent

   ```bash
   uv run adk run multi_tools_agent
   ```

6. Trigger the weather tool

   ```bash
   What is the weather in Singapore right now?
   ```

7. Trigger the search tool

   ```bash
   What are the latest announcements about Google Gemini?
   ```

8. Ask a question needing no tool and confirm none is called

   ```bash
   What is 15 multiplied by 12?
   ```

9. Open adk web and read the function_call and function_response events for each turn

   ```bash
   uv run adk web
   ```


## Test it

The weather question produces a get_weather function_call with a live temperature; the news question produces a tavily_search call; the arithmetic question produces no tool call at all.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
