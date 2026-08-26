"""
main.py
-------
Entry point. Run with:
    python main.py
    python main.py "Which 3 customers have spent the most money overall?"

Runs the agent with a persistent thread_id so it can pause for human
review before any query executes, and prompts you right here in the
terminal to approve, edit, or reject each one.

load_dotenv() picks up .env — where LANGSMITH_TRACING and
LANGSMITH_API_KEY live. Once set, every run is automatically traced in
LangSmith with no other code changes.
"""

import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from agent_graph import build_agent


def prompt_for_decision(query: str) -> dict:
    """Show the pending query and turn a terminal choice into the decision
    dict human_review() is expecting on the other side of interrupt()."""
    print("\n--- Human review requested ---")
    print(f"Proposed query:\n{query}\n")
    choice = input("Approve / Edit / Reject? [a/e/r] (default a): ").strip().lower()

    if choice == "e":
        edited = input("Enter the corrected query: ").strip()
        return {"action": "edit", "query": edited}
    if choice == "r":
        reason = input("Reason (optional): ").strip()
        return {"action": "reject", "reason": reason}
    return {"action": "approve"}


def run(agent, question: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    stream_input = {"messages": [{"role": "user", "content": question}]}

    # Loop because a rejected query sends the agent back to generate_query,
    # which may produce ANOTHER query needing review — so there can be
    # several pause/resume rounds before the graph finally reaches END.
    while True:
        interrupt_payload = None
        for step in agent.stream(stream_input, config=config, stream_mode="values"):
            if "__interrupt__" in step:
                interrupt_payload = step["__interrupt__"][0].value
                break
            step["messages"][-1].pretty_print()

        if interrupt_payload is None:
            break  # graph reached END — nothing left to resume

        decision = prompt_for_decision(interrupt_payload["query"])
        stream_input = Command(resume=decision)


def main():
    agent = build_agent()
    default_question = "Which genre on average has the longest tracks?"
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_question

    print(f"Question: {question}\n")
    print("-" * 60)

    run(agent, question, thread_id=str(uuid.uuid4()))


if __name__ == "__main__":
    main()