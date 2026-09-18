"""
web_search.py
-------------
Web search + synthesis, completely independent of the SQL agent graph in
agent_graph.py / graph_nodes.py.

Deliberately kept as a plain function pipeline (search -> synthesize ->
return) rather than a LangGraph graph: there's no branching, no multi-step
tool loop, and no human-in-the-loop step here, so a graph would just add
ceremony. This is also what makes it easy to swap later — e.g. replace
Tavily with another provider, or drop in caching — without touching
anything in the SQL agent.

Flow:
  1. search_web()      - call Tavily, get up to `max_results` results
  2. answer_from_web()  - hand those results to the LLM to synthesize into
                          one clear, structured answer, and package the
                          sources separately for the caller to display
"""

import os

from tavily import TavilyClient

from config import get_llm

DEFAULT_MAX_RESULTS = 5

_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    """Lazy singleton — built on first use, not at import time, so importing
    this module doesn't blow up just because TAVILY_API_KEY isn't set yet
    (e.g. during tests that never call the web-search path)."""
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file — "
                "get a key at https://app.tavily.com"
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    """Run a Tavily search and return up to `max_results` results, each a
    dict with (at least) 'title', 'url', and 'content' — 'content' is a
    relevance-ranked snippet Tavily already extracted for us, not the raw
    page, which is exactly what we want to hand to the synthesis prompt."""
    client = _get_tavily_client()
    response = client.search(query=query, max_results=max_results, search_depth="advanced")
    return response.get("results", []) or []


SYNTHESIS_PROMPT = """You are a research assistant. You have been given web \
search results for the user's question below. Write a clear, well-structured \
answer using ONLY the information in these results — do not rely on outside \
knowledge, and do not invent anything that isn't supported by them.

If the results genuinely don't contain enough information to answer the \
question, say so plainly instead of guessing.

Format the answer in Markdown (short paragraphs and/or bullet points, \
whichever fits the content best). Do NOT include a sources list or citation \
markers yourself — the calling application shows sources separately.

Question: {question}

Search results:
{formatted_results}
"""


def _format_results_for_prompt(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r.get('title') or 'Untitled'}\n"
            f"URL: {r.get('url') or ''}\n"
            f"{r.get('content') or ''}"
        )
    return "\n\n".join(blocks)


def answer_from_web(question: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict:
    """Run the full web-search pipeline and return a dict:
        {
            "answer": str,                                # synthesized answer
            "sources": [{"title": str, "url": str}, ...],  # for display
            "had_results": bool,                           # False = nothing usable came back
        }
    Never raises for "no results" — that's a normal, expected case handled
    gracefully. DOES raise if Tavily itself errors (bad/missing API key,
    network failure) — the caller (api.py) is responsible for turning that
    into an SSE "error" event instead of a 500.
    """
    results = search_web(question, max_results=max_results)

    if not results:
        return {
            "answer": (
                "I couldn't find any relevant web results for that question. "
                "Try rephrasing it, or being more specific."
            ),
            "sources": [],
            "had_results": False,
        }

    llm = get_llm()
    prompt = SYNTHESIS_PROMPT.format(
        question=question, formatted_results=_format_results_for_prompt(results)
    )
    response = llm.invoke(prompt)

    sources = [
        {"title": r.get("title") or r.get("url") or "Untitled", "url": r["url"]}
        for r in results
        if r.get("url")
    ]

    return {"answer": response.content, "sources": sources, "had_results": True}
