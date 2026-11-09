"""Tools the agent can call: plain Python functions plus JSON schemas that describe them to the LLM."""

import json
from datetime import datetime

from config import DATA_DIR


def _append(filename: str, line: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / filename, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{line}\n")


def record_email(email: str, name: str = "not provided", notes: str = "") -> str:
    print(f"[tool] record_email: {email}")
    _append("contacts.tsv", f"{email}\t{name}\t{notes}")
    return "Email recorded"


def record_unknown_question(question: str) -> str:
    print(f"[tool] record_unknown_question: {question}")
    _append("unknown_questions.tsv", question)
    return "Question recorded"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_email",
            "description": "Record that a user wants to get in touch and has provided their email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email address"},
                    "name": {"type": "string", "description": "The user's name, if provided"},
                    "notes": {"type": "string", "description": "Any useful context about the conversation"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record any question you could not answer because the answer is not in your context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question that couldn't be answered"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]

# Maps the tool name the LLM asks for to the Python function that implements it.
TOOL_REGISTRY = {
    "record_email": record_email,
    "record_unknown_question": record_unknown_question,
}


def run_tool_calls(tool_calls) -> list[dict]:
    """Execute each tool call the model requested and return the results as 'tool' messages."""
    results = []
    for call in tool_calls:
        fn = TOOL_REGISTRY.get(call.function.name)
        if fn is None:
            output = f"Unknown tool: {call.function.name}"
        else:
            try:
                output = fn(**json.loads(call.function.arguments or "{}"))
            except Exception as e:  # report errors back to the model instead of crashing the chat
                output = f"Tool error: {e}"
        results.append({"role": "tool", "content": str(output), "tool_call_id": call.id})
    return results
