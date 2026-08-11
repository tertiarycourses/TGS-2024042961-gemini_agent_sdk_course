"""Lab 1 — verify your Gemini ADK environment.

Run from the labs/ folder:   uv run python lab01/verify_setup.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the labs/ root (one .env for all labs)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def check(label, ok, hint=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok and hint:
        print(f"         -> {hint}")
    return ok


def main():
    print("Gemini ADK environment check\n")
    results = []

    results.append(check(
        f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.13+)",
        sys.version_info >= (3, 13),
        "Install Python 3.13 or later, then re-run uv sync.",
    ))

    try:
        import google.adk  # noqa: F401
        results.append(check("google-adk is installed", True))
    except ImportError:
        results.append(check("google-adk is installed", False,
                             "Run: uv sync"))

    results.append(check(
        f".env found at {env_path}", env_path.exists(),
        "Copy .env.example to .env in the labs/ folder.",
    ))

    key = os.getenv("GOOGLE_API_KEY", "")
    results.append(check(
        "GOOGLE_API_KEY is set", bool(key) and key != "your-google-api-key",
        "Get a free key at https://aistudio.google.com and paste it into .env.",
    ))

    print()
    if all(results):
        print("All checks passed — you are ready for Lab 2.")
    else:
        print("Some checks failed. Fix the items marked FAIL above, then re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
