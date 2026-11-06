"""Configuration: environment variables, paths and the LLM client."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

ROOT = Path(__file__).parent
PROFILE_DIR = ROOT / "me"
DATA_DIR = ROOT / "data"

# Gemini exposes an OpenAI-compatible endpoint, so the standard OpenAI SDK works as-is.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

# Safety net so a misbehaving model can't loop on tool calls forever.
MAX_TOOL_ROUNDS = 5


def get_client() -> OpenAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
