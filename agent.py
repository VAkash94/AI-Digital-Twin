"""The agent: a hand-written tool-calling loop. No agent framework."""

from config import MAX_TOOL_ROUNDS, MODEL, get_client
from prompts import build_system_prompt, load_profile_text, load_summary
from tools import TOOLS, run_tool_calls


class DigitalTwin:
    def __init__(self, model: str = MODEL):
        self.client = get_client()
        self.model = model
        self.system_prompt = build_system_prompt(load_summary(), load_profile_text())

    def _run_agent_loop(self, messages: list[dict]) -> str:
        """Call the LLM; while it asks for tools, run them and feed the results back."""
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)

        rounds = 0
        while response.choices[0].finish_reason == "tool_calls" and rounds < MAX_TOOL_ROUNDS:
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            messages.extend(run_tool_calls(assistant_message.tool_calls))
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)
            rounds += 1

        return response.choices[0].message.content or ""

    def chat(self, message: str, history: list[dict]) -> str:
        # Gradio history items can carry extra keys; the API only wants role + content.
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [{"role": "system", "content": self.system_prompt}, *history, {"role": "user", "content": message}]
        return self._run_agent_loop(messages)
