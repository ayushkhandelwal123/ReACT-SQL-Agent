"""
graph_nodes.py
--------------
Each function here is one node in the agent's graph. Splitting the agent
into dedicated nodes (instead of one big system-prompt-driven loop) buys us
two things:
  1. We can FORCE a specific tool call at a specific point (e.g. always list
     tables before doing anything else) instead of hoping the model does it.
  2. Each step gets its own tailored prompt.

Flow: list_tables -> call_get_schema -> get_schema -> generate_query
      -> (check_query -> human_review -> run_query -> generate_query) loop
      until done. human_review pauses for approval before anything runs.
"""

from typing import Literal

from langchain.messages import AIMessage, ToolMessage
from langgraph.graph import END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from config import get_llm
from db_tools import TOOLS

llm = get_llm()

# Pull out the individual tools by name so each node can bind exactly
# the ones it needs (not the full toolset every time).
list_tables_tool = next(t for t in TOOLS if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in TOOLS if t.name == "sql_db_schema")
run_query_tool = next(t for t in TOOLS if t.name == "sql_db_query")

# ToolNode is a LangGraph prebuilt: given a message with tool_calls, it runs
# the matching tool(s) and appends the results as ToolMessages.
get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_node = ToolNode([run_query_tool], name="run_query")


def _force_tool_call(llm_with_tools, messages, tool_name: str, max_attempts: int = 3):
    """Invoke the model and make sure it actually calls `tool_name`.

    langchain_ollama's ChatOllama.bind_tools() currently ignores the
    tool_choice parameter entirely (confirmed in LangChain's own reference
    docs) — so passing tool_choice="any" does NOT force anything with
    Ollama models. The model is free to respond with plain text instead of
    a tool call, and on harder questions gpt-oss:120b-cloud sometimes does
    exactly that (reasons in prose instead of calling the tool).

    This retries with an explicit nudge until a tool call actually happens,
    instead of silently letting the graph drift forward with no schema /
    no query. Raises after max_attempts so a persistent failure is loud,
    not a silent no-op three nodes later.
    """
    current_messages = list(messages)
    for attempt in range(max_attempts):
        response = llm_with_tools.invoke(current_messages)
        if response.tool_calls:
            return response
        current_messages = current_messages + [
            response,
            {
                "role": "user",
                "content": (
                    f"You must call the {tool_name} tool now — do not respond "
                    "with plain text or explanation."
                ),
            },
        ]
    raise RuntimeError(
        f"Model failed to call '{tool_name}' after {max_attempts} attempts. "
        "Last response was plain text instead of a tool call."
    )


def list_tables(state: MessagesState):
    """Node 1 — no LLM call needed here. We already know we always want the
    table list first, so we construct the tool call by hand and invoke it
    directly. Saves a model round-trip."""
    tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}


def call_get_schema(state: MessagesState):
    """Node 2 — must produce a call to sql_db_schema. tool_choice='any' is
    passed for documentation/forward-compat purposes but is currently a
    no-op with Ollama (see _force_tool_call docstring), so the retry helper
    is what's actually doing the enforcing here."""
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = _force_tool_call(llm_with_tools, state["messages"], "sql_db_schema")
    return {"messages": [response]}


GENERATE_QUERY_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit
your query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""


def generate_query(state: MessagesState):
    """Node 3 — here we do NOT force a tool call. The model is free to either
    write a query (tool call) or, if it already has enough info from earlier
    steps, answer the user directly. This is also the node the graph loops
    back to after every query execution, until the model stops calling tools."""
    system_message = {"role": "system", "content": GENERATE_QUERY_PROMPT}
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


CHECK_QUERY_PROMPT = """
You are a SQL expert with a strong attention to detail.
Double check the sqlite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no
mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""


def check_query(state: MessagesState):
    """Node 4 — a second, independent LLM call whose only job is to critique
    the query generate_query just wrote, before it touches the database.
    Must end in a tool call to run_query_tool (either the original query or
    a corrected one) — same tool_choice caveat as call_get_schema applies,
    so the retry helper enforces it. This is the 'basic' safety net; the
    human-in-the-loop version you'll add later replaces (or supplements) this
    with an actual human approval step."""
    system_message = {"role": "system", "content": CHECK_QUERY_PROMPT}
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = _force_tool_call(
        llm_with_tools, [system_message, user_message], "sql_db_query"
    )
    response.id = state["messages"][-1].id
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
    """Conditional edge after generate_query: if the model produced a tool
    call (it wants to run a query), route to check_query. If it produced a
    plain text answer instead, the agent is done — go to END."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "check_query"


def human_review(state: MessagesState) -> Command[Literal["run_query", "generate_query"]]:
    """Pause right before the checked query touches the database, and ask a
    human to approve, edit, or reject it.

    interrupt() halts execution HERE. The graph's state is checkpointed and
    control returns to whatever called .stream()/.invoke() — that call
    simply stops producing new output. Nothing continues until the caller
    resumes with Command(resume=<decision>), using the SAME thread_id.

    Critical detail: when resumed, this function re-runs from the top. The
    two lines before interrupt() just re-read state (harmless to repeat).
    Everything that matters — the actual routing decision — happens after
    interrupt() returns the human's decision.
    """
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]

    decision = interrupt({"type": "approval_request", "query": query})
    action = decision.get("action", "approve")

    if action == "reject":
        reason = decision.get("reason") or "Rejected by reviewer."
        reject_message = ToolMessage(
            content=(
                f"Query rejected by human reviewer: {reason}. "
                "Write a different query, or ask the user a clarifying "
                "question instead of querying."
            ),
            tool_call_id=tool_call["id"],
        )
        return Command(goto="generate_query", update={"messages": [reject_message]})

    if action == "edit":
        # Mutate the existing tool call in place, then re-submit it under
        # the SAME message id — MessagesState's reducer treats that as an
        # update to the existing message rather than a new one appended.
        last_message.tool_calls[0]["args"]["query"] = decision["query"]
        return Command(goto="run_query", update={"messages": [last_message]})

    # action == "approve" (the default if the caller sends nothing else)
    return Command(goto="run_query")