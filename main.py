"""
main.py
-------
Entry point. Run with:
    python main.py
    python main.py "Which 3 customers have spent the most money overall?"

Asks the agent a question about the chinook.db music store database and
prints each step of its reasoning as it happens.

load_dotenv() picks up .env — that's where LANGSMITH_TRACING and
LANGSMITH_API_KEY live. Once those are set, every run of this script is
automatically traced in LangSmith with zero other code changes: LangChain
reads those env vars itself and wraps every LLM/tool call.
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from agent_graph import build_agent


def main():
    agent = build_agent()

    default_question = "Which genre on average has the longest tracks?"
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_question

    print(f"Question: {question}\n")
    print("-" * 60)

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        # Each `step` is the full state after a node runs; the last message
        # is the newest thing that happened (a tool call, a tool result, or
        # the final answer).
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()