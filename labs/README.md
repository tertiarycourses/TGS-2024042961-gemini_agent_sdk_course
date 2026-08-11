# Gemini Agent ADK — Hands-On Labs

Lab environment for the WSQ course **Develop Multi AI Agent Applications with Gemini Agent ADK**
(TGS-2024042961), built on [Google's Agent Development Kit](https://google.github.io/adk-docs/).

Every lab is a **self-contained folder** — `lab01` … `lab18` — holding that lab's agent script,
any data files it needs, and a `README.md` lab sheet. See [LABS.md](LABS.md) for the full index.

## Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) package manager
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com)

## Setup (once)

```bash
cd labs
uv sync                 # creates .venv and installs google-adk + all dependencies
cp .env.example .env    # then paste your GOOGLE_API_KEY into .env
```

There is **one `.env` for all labs**, in this folder. Every lab loads it automatically.
It is git-ignored — never commit your keys.

```env
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=your-google-api-key
OPENWEATHER_API_KEY=your-openweather-key   # optional — lab03, lab05, lab06, lab10
TAVILY_API_KEY=your-tavily-key             # optional — lab03, lab05, lab06, lab10
OPENAI_API_KEY=your-openai-key             # optional — lab04 only
```

Verify your setup:

```bash
uv run python lab01/verify_setup.py
```

## Running a lab

```bash
uv run adk run lab02    # terminal chat with that lab's agent
uv run adk web          # browser IDE at http://localhost:8000 — pick any lab
```

Two labs are run differently:

```bash
uv run streamlit run lab17/app.py   # lab17 is a Streamlit web app
uv run python lab05/agent.py        # labs 05, 06, 12 are scripts with their own main()
```

## The labs

| # | Folder | Lab | Topic |
|---|---|---|---|
| 1 | `lab01` | Set Up the Gemini ADK Environment and Get an API Key | 1 |
| 2 | `lab02` | Build Your First ADK Agent — A Retail Banking Assistant | 1 |
| 3 | `lab03` | Give an Agent Tools — Live Weather and Web Search | 1 |
| 4 | `lab04` | Swap the Model — Running an ADK Agent on a Non-Gemini LLM | 1 |
| 5 | `lab05` | Give an Agent Memory — Sessions, State and the Runner | 2 |
| 6 | `lab06` | Inspect the Agent Loop — Events, Tool Calls and Final Responses | 2 |
| 7 | `lab07` | Multi-Agent Handoff — Joke Generator to Translator | 2 |
| 8 | `lab08` | Hierarchical Multi-Agent System — The Tutor Agent | 2 |
| 9 | `lab09` | Sequential Workflow Agent — Singapore Transport Route Planner | 2 |
| 10 | `lab10` | Add a Guardrail — Blocking Unsafe Requests with a Callback | 2 |
| 11 | `lab11` | Structured Output — Forcing Valid JSON with Pydantic | 2 |
| 12 | `lab12` | Connect External Tools with MCP — StreamableHTTP and SSE | 2 |
| 13 | `lab13` | Load, Split and Embed Documents into a Vector Store | 3 |
| 14 | `lab14` | Build the Agentic RAG Agent — Retrieval as a Tool | 3 |
| 15 | `lab15` | Evaluate RAG Performance — Retrieval Quality and Groundedness | 3 |
| 16 | `lab16` | Declarative Agents — Configuring a Multi-Agent System in YAML | 4 |
| 17 | `lab17` | Ship the Agent as a Web App with Streamlit | 4 |
| 18 | `lab18` | Capstone — Design, Build and Assess Your Own Multi-Agent Application | 4 |

## Key ADK concepts

| Concept | Where you meet it |
|---|---|
| `Agent` — model, name, description, instruction | lab02 |
| Function tools (docstring + type hints as the contract) | lab03 |
| `LiteLlm` — running on a non-Gemini model | lab04 |
| `Session` + `SessionService` + `Runner` | lab05 |
| Events: `function_call`, `function_response`, final response | lab06 |
| `sub_agents` and `transfer_to_agent` handoff | lab07, lab08 |
| `SequentialAgent` workflows | lab09 |
| `before_model_callback` guardrails | lab10 |
| Pydantic `output_schema` | lab11 |
| Model Context Protocol (`McpToolset`) | lab12 |
| RAG: chunk → embed → store → retrieve | lab13, lab14, lab15 |
| YAML agent configuration | lab16 |
| Streamlit front end for an agent | lab17 |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: google.adk` | Run `uv sync` from the `labs/` folder |
| Authentication / 401 errors | Check `.env` exists in `labs/` and `GOOGLE_API_KEY` has no quotes or trailing spaces |
| `adk` command not found | Prefix with `uv run`, e.g. `uv run adk web` |
| A lab does not appear in `adk web` | Run `adk web` from `labs/`, not from inside a lab folder |
| RAG lab returns nothing | Delete `lab13/chroma_db` (or `lab14`/`lab15`) and re-run to re-index |

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google AI Studio](https://aistudio.google.com) — free Gemini API keys

---

© 2026 Tertiary Infotech Academy Pte Ltd. All rights reserved.
