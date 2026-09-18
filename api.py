"""
api.py
------
FastAPI wrapper around the same agent from agent_graph.py.

Run with:
    uvicorn api:app --reload

One agent + one checkpointer is created ONCE at import time (module-level
`agent`, below) and reused across every request. That single shared
checkpointer is what lets a conversation paused for human review in one
HTTP call get resumed correctly by a LATER call, as long as both carry the
same thread_id — each request is otherwise stateless, same as any web API.

Endpoints:
  POST /ask                 - ask a new question, wait for the full result
  POST /review/{thread_id}  - approve / edit / reject a pending query, wait
                               for the full result
  POST /ask/stream          - same as /ask, but streamed as Server-Sent
                               Events so the frontend can show each agent
                               step as it happens
  POST /review/{thread_id}/stream - streamed version of /review
  GET  /health               - liveness check
"""

import json
import uuid
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.types import Command

from agent_graph import build_agent
from web_search import answer_from_web

app = FastAPI(title="Chinook SQL Agent API")

# Wide-open CORS for LOCAL DEV ONLY. A React dev server on localhost:5173
# is a different origin than this API's localhost:8000, and browsers block
# cross-origin requests by default unless the server explicitly allows it.
# Tighten allow_origins to your real frontend URL before deploying anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at import time, not per-request — this is what makes pause/
# resume across separate HTTP calls possible.
agent = build_agent()


class AskRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    query: Optional[str] = None
    reason: Optional[str] = None


class AgentResponse(BaseModel):
    thread_id: str
    status: Literal["pending_review", "done"]
    answer: Optional[str] = None
    pending_query: Optional[str] = None


def _run_until_pause(stream_input, thread_id: str) -> AgentResponse:
    """Drive the graph forward until it either pauses for review or
    reaches END, and translate whichever happened into an API response."""
    config = {"configurable": {"thread_id": thread_id}}
    for step in agent.stream(stream_input, config=config, stream_mode="values"):
        if "__interrupt__" in step:
            payload = step["__interrupt__"][0].value
            return AgentResponse(
                thread_id=thread_id,
                status="pending_review",
                pending_query=payload["query"],
            )

    final_state = agent.get_state(config)
    last_message = final_state.values["messages"][-1]
    return AgentResponse(thread_id=thread_id, status="done", answer=last_message.content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AgentResponse)
def ask(request: AskRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    stream_input = {"messages": [{"role": "user", "content": request.question}]}
    return _run_until_pause(stream_input, thread_id)


@app.post("/review/{thread_id}", response_model=AgentResponse)
def review(thread_id: str, request: ReviewRequest):
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if not state.next:
        raise HTTPException(
            status_code=400,
            detail="No pending review for this thread_id (already finished, or unknown).",
        )

    decision: dict = {"action": request.action}
    if request.action == "edit":
        if not request.query:
            raise HTTPException(status_code=400, detail="query is required when action='edit'")
        decision["query"] = request.query
    if request.action == "reject" and request.reason:
        decision["reason"] = request.reason

    return _run_until_pause(Command(resume=decision), thread_id)


def _sse(data: dict) -> str:
    """Format one Server-Sent Event. Two trailing newlines is the SSE
    wire-format delimiter that marks the end of an event."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def _message_to_event_fields(msg) -> dict:
    return {
        "role": msg.type,  # "human" | "ai" | "tool"
        "content": msg.content,
        "tool_calls": getattr(msg, "tool_calls", None) or [],
        "name": getattr(msg, "name", None),
    }


def _stream_graph(stream_input, thread_id: str):
    """Generator that drives the graph forward and yields one SSE event per
    node update, using stream_mode='updates' so each event is tagged with
    exactly which node produced it (values mode only gives full state
    snapshots, with no way to tell which node just ran).
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for update in agent.stream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                payload = update["__interrupt__"][0].value
                yield _sse(
                    {"event": "interrupt", "thread_id": thread_id, "query": payload["query"]}
                )
                return

            for node_name, node_output in update.items():
                # A Command(goto=...) with no `update=` (e.g. a plain
                # approve) contributes no state change, so node_output is
                # None here — nothing to stream for that node.
                if not node_output:
                    continue
                for msg in node_output.get("messages", []):
                    yield _sse(
                        {
                            "event": "step",
                            "thread_id": thread_id,
                            "node": node_name,
                            **_message_to_event_fields(msg),
                        }
                    )

        # Loop finished without ever hitting an interrupt -> graph reached END.
        final_state = agent.get_state(config)
        last_message = final_state.values["messages"][-1]
        yield _sse({"event": "done", "thread_id": thread_id, "answer": last_message.content})
    except Exception as e:
        yield _sse({"event": "error", "thread_id": thread_id, "detail": str(e)})


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    stream_input = {"messages": [{"role": "user", "content": request.question}]}
    return StreamingResponse(_stream_graph(stream_input, thread_id), media_type="text/event-stream")


@app.post("/review/{thread_id}/stream")
def review_stream(thread_id: str, request: ReviewRequest):
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if not state.next:
        raise HTTPException(
            status_code=400,
            detail="No pending review for this thread_id (already finished, or unknown).",
        )

    decision: dict = {"action": request.action}
    if request.action == "edit":
        if not request.query:
            raise HTTPException(status_code=400, detail="query is required when action='edit'")
        decision["query"] = request.query
    if request.action == "reject" and request.reason:
        decision["reason"] = request.reason

    return StreamingResponse(
        _stream_graph(Command(resume=decision), thread_id), media_type="text/event-stream"
    )


# ---------------------------------------------------------------------
# Web search — independent of the SQL agent above. No thread_id/checkpoint
# involved: each question is a single, stateless search + synthesize call,
# reusing web_search.py so this endpoint stays a thin adapter over it.
# ---------------------------------------------------------------------


class WebSearchRequest(BaseModel):
    question: str


def _stream_web_search(question: str):
    """Two SSE events: one 'step' so the frontend can show a searching
    indicator (reusing the same chain UI as the SQL agent), then 'done'
    with the synthesized answer and sources — or 'error' if Tavily/the LLM
    call itself fails (missing API key, network issue, etc.)."""
    request_id = str(uuid.uuid4())
    try:
        yield _sse(
            {
                "event": "step",
                "thread_id": request_id,
                "node": "web_search",
                "role": "tool",
                "content": f'Searching the web for: "{question}"',
                "tool_calls": [],
            }
        )

        result = answer_from_web(question)

        yield _sse(
            {
                "event": "done",
                "thread_id": request_id,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )
    except Exception as e:
        yield _sse({"event": "error", "thread_id": request_id, "detail": str(e)})


@app.post("/websearch/stream")
def websearch_stream(request: WebSearchRequest):
    return StreamingResponse(
        _stream_web_search(request.question), media_type="text/event-stream"
    )
