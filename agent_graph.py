"""
agent_graph.py
--------------
Wires the node functions from graph_nodes.py into a StateGraph and compiles
it into a runnable agent.

Includes a checkpointer. This is required for the human_review step:
interrupt()/Command(resume=...) only work when LangGraph can persist and
reload the paused state — that persistence IS the checkpointer.
InMemorySaver is fine for local dev (state lives only as long as the
process runs). For production you'd swap in SqliteSaver or PostgresSaver
so a paused conversation survives a restart.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from graph_nodes import (
    list_tables,
    call_get_schema,
    get_schema_node,
    generate_query,
    check_query,
    human_review,
    run_query_node,
    should_continue,
)


def build_agent(checkpointer=None):
    """checkpointer defaults to a fresh InMemorySaver if you don't pass one.
    The FastAPI backend passes its OWN single shared instance so that a
    thread paused in one HTTP request is still there for the next one —
    a script that just wants to run once can ignore this and get a working
    default.
    """
    builder = StateGraph(MessagesState)

    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(human_review)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")

    builder.add_conditional_edges("generate_query", should_continue)

    builder.add_edge("check_query", "human_review")
    # human_review returns Command(goto=...) itself — either "run_query" on
    # approve/edit, or back to "generate_query" on reject — so no static
    # edge is declared from it here. The Literal in its return type
    # annotation is what tells LangGraph the possible destinations.
    builder.add_edge("run_query", "generate_query")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == "__main__":
    agent = build_agent()
    print("Graph compiled successfully. Nodes:", list(agent.get_graph().nodes.keys()))

    try:
        import pathlib
        pathlib.Path("graph.png").write_bytes(agent.get_graph().draw_mermaid_png())
        print("Saved graph.png")
    except Exception as e:
        print(f"Skipped graph visualization: {e}")
