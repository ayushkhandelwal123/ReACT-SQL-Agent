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
  POST /ask                 - ask a new question (creates a thread_id if
                               you don't supply one)
  POST /review/{thread_id}  - approve / edit / reject a pending query and
                               continue
  GET  /health               - liveness check
"""

import uuid
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from agent_graph import build_agent

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