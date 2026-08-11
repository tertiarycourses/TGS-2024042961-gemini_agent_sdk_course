# Lab 11 — Structured Output — Forcing Valid JSON with Pydantic

**Topic 02:** Build A Multi Agent App with Gemini ADK  
**Learning outcome:** LO3 — produce machine-readable agent output for downstream systems.  
**Agent folder:** `labs/agent_structured_output`  
**Tools:** google-adk, Pydantic BaseModel, output_schema

## Goal

Attach a Pydantic output_schema so the agent returns a validated Recipe object with typed fields instead of free-form prose that a downstream system cannot parse.

## What you'll build

agent_structured_output — an agent whose every reply is schema-valid JSON.

## Step-by-step

1. Read the Recipe model and the output_schema argument

   ```bash
   cat agent_structured_output/agent.py
   ```

2. Note the typed fields: title, ingredients, cooking_time, servings, instructions
3. Run the agent

   ```bash
   uv run adk run agent_structured_output
   ```

4. Request a recipe

   ```bash
   Chicken rice
   ```

5. Confirm the reply is JSON with all five fields and correct types
6. Add a difficulty field to the Recipe model

   ```bash
   difficulty: str
   ```

7. Re-run and confirm the new field appears in the output
8. Note the ADK restriction that an agent with output_schema cannot also use tools

## Test it

Every response parses as JSON matching the Recipe schema, cooking_time is an integer, and your added difficulty field is populated.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.1 — © 2026 Tertiary Infotech Academy Pte Ltd*
