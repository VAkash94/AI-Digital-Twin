import json

from config import MAX_TOOL_ROUNDS, MODEL, get_client
from prompts import build_evaluator_prompt, build_system_prompt, load_profile_text, load_summary
from tools import TOOLS, run_tool_calls

FALLBACK_REPLY = (
    "Sorry, I'm not able to give a reliable answer to that. "
    "Feel free to leave your email and I'll follow up personally."
)


class DigitalTwin:
    def __init__(self, model: str = MODEL):
        self.client = get_client()
        self.model = model
        self.system_prompt = build_system_prompt(load_summary(), load_profile_text())
        self.evaluator_prompt = build_evaluator_prompt(self.system_prompt)

    # ---- 1. Agent loop -------------------------------------------------------

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

    # ---- 2. Evaluator (guardrail) -------------------------------------------

    def _evaluate(self, user_message: str, reply: str) -> tuple[bool, str]:
        """Ask a second LLM call to judge whether the reply is on-topic and grounded."""
        messages = [
            {"role": "system", "content": self.evaluator_prompt},
            {"role": "user", "content": f"Visitor: {user_message}\n\nTwin's reply: {reply}"},
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            verdict = json.loads(raw)
            return bool(verdict.get("acceptable")), str(verdict.get("feedback", ""))
        except json.JSONDecodeError:
            print(f"[evaluator] could not parse verdict: {raw!r}")
            return False, "Evaluator returned an unreadable verdict."

    # ---- 3. Public entry point ----------------------------------------------

    def chat(self, message: str, history: list[dict]) -> str:
        # Gradio history items can carry extra keys; the API only wants role + content.
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [{"role": "system", "content": self.system_prompt}, *history, {"role": "user", "content": message}]

        reply = self._run_agent_loop(list(messages))
        acceptable, feedback = self._evaluate(message, reply)
        if acceptable:
            return reply

        # One retry: tell the model why its previous answer was rejected.
        print(f"[evaluator] rejected: {feedback}")
        retry_messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"[Quality check] Your previous reply was rejected: {feedback} "
                                        "Please answer the visitor's last message again, fixing this."},
        ]
        reply = self._run_agent_loop(retry_messages)
        acceptable, feedback = self._evaluate(message, reply)
        if acceptable:
            return reply

        print(f"[evaluator] rejected again: {feedback}")
        return FALLBACK_REPLY
