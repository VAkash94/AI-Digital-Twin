# Digital Twin

An AI "digital twin" that chats with website visitors about a person's career, skills and experience.

Built without an agent framework: tool calling, the agent loop and guardrails written by hand in plain Python.

Work in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a [Google AI Studio API key](https://aistudio.google.com/apikey).

```bash
cp .env.example .env      # then add your GOOGLE_API_KEY
uv sync
```
