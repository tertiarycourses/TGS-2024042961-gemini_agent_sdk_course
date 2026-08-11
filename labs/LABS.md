# Develop Multi AI Agent Applications with Gemini Agent ADK — Hands-On Labs

**Course Code:** TGS-2024042961  ·  **18 labs across 4 topics**  ·  **v1.1**

Full step-by-step instructions for every lab are in the **Learner Guide** (`LG-*.md` at the repo root, or the DOCX/PDF in `courseware/`).

| # | Lab | Agent folder | Topic |
|---|---|---|---|
| 1 | [Set Up the Gemini ADK Environment and Get an API Key](guides/lab-01-set-up-the-gemini-adk-environment-and-get-an-api-k.md) | — | 1 |
| 2 | [Build Your First ADK Agent — A Retail Banking Assistant](guides/lab-02-build-your-first-adk-agent---a-retail-banking-assi.md) | `basic_agent` | 1 |
| 3 | [Give an Agent Tools — Live Weather and Web Search](guides/lab-03-give-an-agent-tools---live-weather-and-web-search.md) | `multi_tools_agent` | 1 |
| 4 | [Swap the Model — Running an ADK Agent on a Non-Gemini LLM](guides/lab-04-swap-the-model---running-an-adk-agent-on-a-non-gem.md) | `agent_model` | 1 |
| 5 | [Give an Agent Memory — Sessions, State and the Runner](guides/lab-05-give-an-agent-memory---sessions--state-and-the-run.md) | `agent_session` | 2 |
| 6 | [Inspect the Agent Loop — Events, Tool Calls and Final Responses](guides/lab-06-inspect-the-agent-loop---events--tool-calls-and-fi.md) | `agent_interact` | 2 |
| 7 | [Multi-Agent Handoff — Joke Generator to Translator](guides/lab-07-multi-agent-handoff---joke-generator-to-translator.md) | `agent_handoff` | 2 |
| 8 | [Hierarchical Multi-Agent System — The Tutor Agent](guides/lab-08-hierarchical-multi-agent-system---the-tutor-agent.md) | `tutor_agent` | 2 |
| 9 | [Sequential Workflow Agent — Singapore Transport Route Planner](guides/lab-09-sequential-workflow-agent---singapore-transport-ro.md) | `transport_agent` | 2 |
| 10 | [Add a Guardrail — Blocking Unsafe Requests with a Callback](guides/lab-10-add-a-guardrail---blocking-unsafe-requests-with-a-.md) | `agent_guardrail` | 2 |
| 11 | [Structured Output — Forcing Valid JSON with Pydantic](guides/lab-11-structured-output---forcing-valid-json-with-pydant.md) | `agent_structured_output` | 2 |
| 12 | [Connect External Tools with MCP — StreamableHTTP and SSE](guides/lab-12-connect-external-tools-with-mcp---streamablehttp-a.md) | `agent_mcp` | 2 |
| 13 | [Load, Split and Embed Documents into a Vector Store](guides/lab-13-load--split-and-embed-documents-into-a-vector-stor.md) | `agent_rag` | 3 |
| 14 | [Build the Agentic RAG Agent — Retrieval as a Tool](guides/lab-14-build-the-agentic-rag-agent---retrieval-as-a-tool.md) | `agent_rag` | 3 |
| 15 | [Evaluate RAG Performance — Retrieval Quality and Groundedness](guides/lab-15-evaluate-rag-performance---retrieval-quality-and-g.md) | `agent_rag` | 3 |
| 16 | [Declarative Agents — Configuring a Multi-Agent System in YAML](guides/lab-16-declarative-agents---configuring-a-multi-agent-sys.md) | `transport_agent_yaml` | 4 |
| 17 | [Ship the Agent as a Web App with Streamlit](guides/lab-17-ship-the-agent-as-a-web-app-with-streamlit.md) | `transport_agent_streamlit` | 4 |
| 18 | [Capstone — Design, Build and Assess Your Own Multi-Agent Application](guides/lab-18-capstone---design--build-and-assess-your-own-multi.md) | — | 4 |

## Setup

```bash
cd labs
uv sync
cp .env.example .env    # then paste your keys
```

```bash
uv run adk run <agent_folder>   # terminal
uv run adk web                  # browser IDE
```
