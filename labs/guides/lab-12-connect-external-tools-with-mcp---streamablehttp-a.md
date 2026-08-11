# Lab 12 — Connect External Tools with MCP — StreamableHTTP and SSE

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 / LO2 — integrate third-party tools through the Model Context Protocol.  
**Agent folder:** `labs/agent_mcp`  
**Tools:** google-adk, McpToolset, StreamableHTTPConnectionParams, SseConnectionParams, n8n MCP server

## Goal

Connect an ADK agent to a remote MCP server so it can use tools it does not own, using both the StreamableHTTP and SSE transports.

## What you'll build

agent_mcp and agent_mcp_sse — agents whose toolset is discovered at runtime from an MCP server.

## Step-by-step

1. Read how McpToolset discovers tools from the server URL

   ```bash
   cat agent_mcp/agent.py
   ```

2. Run the StreamableHTTP MCP agent and read the printed tool list

   ```bash
   uv run python agent_mcp/agent.py
   ```

3. Send a request that exercises one of the discovered tools
4. Compare the SSE variant and note that only the connection params differ

   ```bash
   cat agent_mcp_sse/agent.py
   ```

5. Explain when SSE is preferred over StreamableHTTP
6. Point the MCP_SERVER_URL at a different MCP server and re-run
7. Confirm the agent's available tools change without any change to the agent code

## Test it

The agent prints the tools discovered from the MCP server and successfully invokes one; changing the server URL changes the toolset with no code edit.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
