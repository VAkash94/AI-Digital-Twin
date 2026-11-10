"""Gradio chat UI for the digital twin. Run with:  uv run app.py"""

import gradio as gr

from agent import DigitalTwin

twin = DigitalTwin()

EXAMPLES = [
    "Tell me about your background.",
    "What are your strongest technical skills?",
    "What kind of projects have you worked on?",
    "I'd like to get in touch with you.",
]

if __name__ == "__main__":
    gr.ChatInterface(
        twin.chat,
        title="Digital Twin",
        description="Chat with my AI twin about my career, skills and experience.",
        examples=EXAMPLES,
    ).launch(inbrowser=True)
