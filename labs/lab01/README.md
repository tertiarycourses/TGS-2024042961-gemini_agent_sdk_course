# Lab 1 — Set Up the Gemini ADK Environment and Get an API Key

**Topic 01:** Overview of Agentic AI in Gemini ADK  
**Learning outcome:** LO1 / LO2 — establish a working Gemini ADK development environment.  
**Tools:** Google AI Studio, uv, Python 3.13, google-adk, python-dotenv

## Goal

Install the Agent Development Kit toolchain with uv, obtain a free Google AI Studio API key, and store it safely in a .env file so every agent in the course can authenticate.

## What you'll build

A working Python 3.13 project with google-adk installed and a validated GOOGLE_API_KEY.

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
uv run adk run lab01     # terminal chat
uv run adk web              # browser IDE, then pick lab01 at http://localhost:8000
```

## Step-by-step

1. Clone the course lab repository

   ```bash
   git clone https://github.com/tertiarycourses/TGS-2024042961-Develop-Multi-AI-Agent-Applications-with-Gemini-Agent-ADK.git
   ```

2. Change into the labs folder

   ```bash
   cd TGS-2024042961-Develop-Multi-AI-Agent-Applications-with-Gemini-Agent-ADK/labs
   ```

3. Install all dependencies with uv (creates the .venv automatically)

   ```bash
   uv sync
   ```

4. Open Google AI Studio and sign in with your Google account, then click Get API key → Create API key
5. Create a .env file in the labs folder and paste your key

   ```bash
   cat > .env <<'EOF'
   GOOGLE_GENAI_USE_VERTEXAI=0
   GOOGLE_API_KEY=your-google-api-key
   OPENWEATHER_API_KEY=your-openweather-key
   TAVILY_API_KEY=your-tavily-key
   EOF
   ```

6. Confirm the ADK command-line tool is on the path

   ```bash
   uv run adk --help
   ```


## Test it

uv run adk --help prints the ADK usage banner listing the run, web and eval sub-commands, and no ModuleNotFoundError is raised.

---

*Develop Multi AI Agent Applications with Gemini Agent ADK (TGS-2024042961) v1.2 — © 2026 Tertiary Infotech Academy Pte Ltd*
