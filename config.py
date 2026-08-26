"""
config.py
---------
Single place that decides which LLM the agent uses.

Every other module imports get_llm() from here instead of constructing
its own model. If you ever want to swap gpt-oss:120b-cloud for another
Ollama cloud model (or a different provider entirely), this is the only
file you touch.

Prerequisite: you must be signed in to Ollama cloud once, from a terminal:
    ollama signin
Then the cloud model is addressable by name, same as a local one.
"""

from langchain_ollama import ChatOllama

MODEL_NAME = "gpt-oss:120b-cloud"


def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Return a configured chat model.

    temperature=0.0 because this is a SQL-generation agent — we want
    deterministic, repeatable queries, not creative variation.
    """
    return ChatOllama(model=MODEL_NAME, temperature=temperature)
