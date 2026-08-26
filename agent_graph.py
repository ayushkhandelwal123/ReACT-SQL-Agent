"""
agent_graph.py
--------------
Wires the node functions from graph_nodes.py into a StateGraph and compiles
it into a runnable agent. This is the file that defines the *shape* of the
agent's workflow — nodes are the steps, edges are "what happens next".
"""

from langgraph.graph import START, MessagesState, StateGraph

from graph_nodes import (
    list_tables,
    call_get_schema,
    get_schema_node,
    generate_query,
    check_query,
    run_query_node,
    should_continue,
)


def build_agent():
    builder = StateGraph(MessagesState)

    # Register every node. add_node(fn) uses the function's own name as the
    # node id; add_node(fn, "name") lets us rename prebuilt ToolNodes.
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    # Fixed path: always list tables, then get schema, then try to answer.
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")

    # Branch point: generate_query decides "done" vs "need to run a query".
    builder.add_conditional_edges("generate_query", should_continue)

    # The check -> run -> generate loop: every query gets self-reviewed,
    # executed, and the result is fed back so the model can decide if it
    # has enough to answer or needs to query again.
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile()


if __name__ == "__main__":
    # Quick structural check: compiles the graph and (optionally) draws it.
    # `python agent_graph.py` on its own just confirms the wiring is valid.
    agent = build_agent()
    print("Graph compiled successfully. Nodes:", list(agent.get_graph().nodes.keys()))

    try:
        import pathlib
        pathlib.Path("graph.png").write_bytes(agent.get_graph().draw_mermaid_png())
        print("Saved graph.png")
    except Exception as e:
        # This calls out to mermaid.ink over the network — fine to skip if
        # it fails (e.g. no internet, or a Windows SSL quirk).
        print(f"Skipped graph visualization: {e}")
