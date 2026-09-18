import { useEffect, useRef, useState } from "react";
import { askStream, reviewStream, webSearchStream } from "./api";
import "./styles.css";

// Maps a graph node name to the short label shown as each channel's eyebrow.
// This is real pipeline information (which stage produced this message),
// not decoration.
const STAGE_LABELS = {
  list_tables: "Tables",
  call_get_schema: "Schema",
  get_schema: "Schema",
  generate_query: "Query",
  check_query: "Check",
  human_review: "Review",
  run_query: "Result",
  web_search: "Search",
};

const STATUS_TEXT = {
  idle: "Idle",
  running: "Running",
  awaiting_review: "Awaiting review",
  done: "Done",
  error: "Error",
};

function newTurn(question, mode) {
  return {
    question,
    mode,
    steps: [],
    pendingQuery: null,
    editedQuery: "",
    finalAnswer: null,
    sources: [],
    status: "running",
    error: null,
  };
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("sql"); // "sql" | "web"
  const [turns, setTurns] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  const lastTurn = turns[turns.length - 1] ?? null;
  const overallStatus = lastTurn ? lastTurn.status : "idle";
  const isBusy = overallStatus === "running" || overallStatus === "awaiting_review";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // Applies an update to the turn at `index`. Used instead of a plain
  // "update the last turn" helper so the textarea/review handlers can
  // target a turn by its own index rather than assuming it's last —
  // in practice it always is (only the newest turn is ever interactive),
  // but this keeps that assumption out of the update logic itself.
  function updateTurn(index, updater) {
    setTurns((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      const next = [...prev];
      next[index] = updater(next[index]);
      return next;
    });
  }

  function updateLastTurn(updater) {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      next[next.length - 1] = updater(next[next.length - 1]);
      return next;
    });
  }

  // Returns an event handler bound to a specific mode, captured at the
  // moment the request was sent — NOT read from component state inside
  // the handler, which would risk a stale closure since `turns` may have
  // moved on by the time an async SSE event arrives.
  function makeEventHandler(mode) {
    return (data) => {
      if (data.event === "step") {
        updateLastTurn((t) => ({ ...t, steps: [...t.steps, data] }));
      } else if (data.event === "interrupt") {
        // Only the SQL agent ever emits "interrupt" — web search has no
        // review step — but the mode check keeps that assumption explicit.
        if (mode === "sql") setThreadId(data.thread_id);
        updateLastTurn((t) => ({
          ...t,
          pendingQuery: data.query,
          editedQuery: data.query,
          status: "awaiting_review",
        }));
      } else if (data.event === "done") {
        if (mode === "sql") setThreadId(data.thread_id);
        updateLastTurn((t) => ({
          ...t,
          finalAnswer: data.answer,
          sources: data.sources || [],
          status: "done",
        }));
      } else if (data.event === "error") {
        updateLastTurn((t) => ({ ...t, status: "error", error: data.detail }));
      }
    };
  }

  async function askQuestion() {
    const q = question.trim();
    if (!q || isBusy) return;
    const currentMode = mode;
    setQuestion("");
    setTurns((prev) => [...prev, newTurn(q, currentMode)]);
    const handleEvent = makeEventHandler(currentMode);
    try {
      if (currentMode === "web") {
        await webSearchStream(q, handleEvent);
      } else {
        await askStream(q, threadId, handleEvent);
      }
    } catch (err) {
      updateLastTurn((t) => ({ ...t, status: "error", error: err.message }));
    } finally {
      inputRef.current?.focus();
    }
  }

  async function sendReview(action) {
    if (!threadId || !lastTurn) return;
    const decision = { action };
    if (action === "edit") decision.query = lastTurn.editedQuery;
    updateLastTurn((t) => ({ ...t, pendingQuery: null, status: "running" }));
    try {
      await reviewStream(threadId, decision, makeEventHandler("sql"));
    } catch (err) {
      updateLastTurn((t) => ({ ...t, status: "error", error: err.message }));
    }
  }

  function startNewChat() {
    if (isBusy) return;
    setTurns([]);
    setThreadId(null);
    setQuestion("");
    inputRef.current?.focus();
  }

  return (
    <div className="console">
      <header className="console-header">
        <div className="wordmark">
          <span className="wordmark-main">CHINOOK</span>
          <span className="wordmark-sub">Agent Console</span>
        </div>
        <div className="header-right">
          <div className={`status-chip status-${overallStatus}`}>
            <span className="status-dot" />
            {STATUS_TEXT[overallStatus]}
          </div>
          <button className="new-chat-btn" onClick={startNewChat} disabled={isBusy || turns.length === 0}>
            New chat
          </button>
        </div>
      </header>

      <div className="conversation">
        {turns.length === 0 && (
          <div className="chain-empty">
            Ask about the music store database, or switch to Web Search below for
            current, internet-based information. Follow-ups in Database mode keep
            the same context — no need to repeat yourself.
          </div>
        )}

        {turns.map((turn, ti) => {
          const isLastTurn = ti === turns.length - 1;
          const hasMoreBelow = (i) =>
            i < turn.steps.length - 1 || Boolean(turn.pendingQuery) || Boolean(turn.finalAnswer);

          return (
            <div className="turn" key={ti}>
              <div className="turn-question">
                <p className="turn-question-text">{turn.question}</p>
                <span className="turn-question-tag">
                  {turn.mode === "web" ? "Web search" : "Database"}
                </span>
              </div>

              <div className="chain">
                {turn.steps.map((step, i) => (
                  <div className="chain-row" key={i}>
                    <div className="chain-rail">
                      <span className={`chain-dot dot-${step.role}`} />
                      {hasMoreBelow(i) && <span className="chain-wire" />}
                    </div>
                    <div className={`channel channel-${step.role}`}>
                      <div className="channel-eyebrow">{STAGE_LABELS[step.node] || step.node}</div>
                      {step.content && <p className="channel-content">{step.content}</p>}
                      {step.tool_calls?.map((tc, j) => (
                        <pre className="channel-code" key={j}>
                          {tc.name}({JSON.stringify(tc.args)})
                        </pre>
                      ))}
                    </div>
                  </div>
                ))}

                {turn.pendingQuery && (
                  <div className="chain-row">
                    <div className="chain-rail">
                      <span className="chain-dot dot-hold" />
                      <span className="chain-wire" />
                    </div>
                    <div className="channel channel-hold">
                      <div className="channel-eyebrow">Hold — review required</div>
                      <textarea
                        className="hold-editor"
                        value={turn.editedQuery}
                        onChange={(e) =>
                          updateTurn(ti, (t) => ({ ...t, editedQuery: e.target.value }))
                        }
                        rows={4}
                        spellCheck={false}
                        disabled={!isLastTurn}
                      />
                      <div className="hold-actions">
                        <button className="btn btn-approve" onClick={() => sendReview("approve")}>
                          Approve
                        </button>
                        <button
                          className="btn btn-edit"
                          onClick={() => sendReview("edit")}
                          disabled={turn.editedQuery.trim() === turn.pendingQuery.trim()}
                        >
                          Approve edited query
                        </button>
                        <button className="btn btn-reject" onClick={() => sendReview("reject")}>
                          Reject
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {turn.finalAnswer && (
                  <div className="chain-row">
                    <div className="chain-rail">
                      <span className="chain-dot dot-answer" />
                    </div>
                    <div className="channel channel-answer">
                      <div className="channel-eyebrow">Answer</div>
                      <p className="channel-content answer-text">{turn.finalAnswer}</p>
                      {turn.sources.length > 0 && (
                        <div className="sources-list">
                          <div className="sources-label">Sources</div>
                          {turn.sources.map((s, si) => (
                            <a
                              key={si}
                              className="source-link"
                              href={s.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {s.title || s.url}
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {turn.status === "error" && turn.error && (
                  <div className="error-banner">{turn.error}</div>
                )}
              </div>
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>

      <div className="mode-toggle" role="tablist" aria-label="Question mode">
        <button
          className={`mode-btn ${mode === "sql" ? "mode-active" : ""}`}
          onClick={() => setMode("sql")}
          role="tab"
          aria-selected={mode === "sql"}
        >
          Database
        </button>
        <button
          className={`mode-btn ${mode === "web" ? "mode-active" : ""}`}
          onClick={() => setMode("web")}
          role="tab"
          aria-selected={mode === "web"}
        >
          Web Search
        </button>
      </div>

      <div className="ask-bar">
        <span className="ask-prompt">&rsaquo;</span>
        <input
          ref={inputRef}
          className="ask-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && askQuestion()}
          placeholder={
            mode === "web"
              ? turns.length === 0
                ? "Search the web for current information…"
                : "Ask another web search…"
              : turns.length === 0
                ? "Ask about the music store database…"
                : "Ask a follow-up…"
          }
          disabled={isBusy}
        />
        <button className="ask-button" onClick={askQuestion} disabled={isBusy || !question.trim()}>
          {isBusy ? "Working…" : "Ask"}
        </button>
      </div>
    </div>
  );
}
