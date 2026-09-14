import { useEffect, useRef, useState } from "react";
import { askStream, reviewStream } from "./api";
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
};

const STATUS_TEXT = {
  idle: "Idle",
  running: "Running",
  awaiting_review: "Awaiting review",
  done: "Done",
  error: "Error",
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [pendingQuery, setPendingQuery] = useState(null);
  const [editedQuery, setEditedQuery] = useState("");
  const [finalAnswer, setFinalAnswer] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [steps, pendingQuery, finalAnswer]);

  function handleEvent(data) {
    if (data.event === "step") {
      setSteps((prev) => [...prev, data]);
    } else if (data.event === "interrupt") {
      setThreadId(data.thread_id);
      setPendingQuery(data.query);
      setEditedQuery(data.query);
      setStatus("awaiting_review");
    } else if (data.event === "done") {
      setThreadId(data.thread_id);
      setFinalAnswer(data.answer);
      setStatus("done");
    } else if (data.event === "error") {
      setError(data.detail);
      setStatus("error");
    }
  }

  async function askQuestion() {
    const q = question.trim();
    if (!q || status === "running") return;
    setSteps([]);
    setPendingQuery(null);
    setFinalAnswer(null);
    setError(null);
    setThreadId(null);
    setStatus("running");
    try {
      await askStream(q, handleEvent);
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  async function sendReview(action) {
    if (!threadId) return;
    setPendingQuery(null);
    setStatus("running");
    const decision = { action };
    if (action === "edit") decision.query = editedQuery;
    try {
      await reviewStream(threadId, decision, handleEvent);
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  const queryWasEdited =
    pendingQuery !== null && editedQuery.trim() !== pendingQuery.trim();
  const hasMoreBelow = (i) => i < steps.length - 1 || Boolean(pendingQuery) || Boolean(finalAnswer);

  return (
    <div className="console">
      <header className="console-header">
        <div className="wordmark">
          <span className="wordmark-main">CHINOOK</span>
          <span className="wordmark-sub">Agent Console</span>
        </div>
        <div className={`status-chip status-${status}`}>
          <span className="status-dot" />
          {STATUS_TEXT[status]}
        </div>
      </header>

      <div className="ask-bar">
        <span className="ask-prompt">&rsaquo;</span>
        <input
          className="ask-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && askQuestion()}
          placeholder="Ask about the music store database…"
          disabled={status === "running"}
        />
        <button
          className="ask-button"
          onClick={askQuestion}
          disabled={status === "running" || !question.trim()}
        >
          {status === "running" ? "Working…" : "Ask"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="chain">
        {steps.length === 0 && !pendingQuery && !finalAnswer && !error && (
          <div className="chain-empty">
            Ask a question to see the agent's reasoning chain run stage by stage.
          </div>
        )}

        {steps.map((step, i) => (
          <div className="chain-row" key={i}>
            <div className="chain-rail">
              <span className={`chain-dot dot-${step.role}`} />
              {hasMoreBelow(i) && <span className="chain-wire" />}
            </div>
            <div className={`channel channel-${step.role}`}>
              <div className="channel-eyebrow">
                {STAGE_LABELS[step.node] || step.node}
              </div>
              {step.content && <p className="channel-content">{step.content}</p>}
              {step.tool_calls?.map((tc, j) => (
                <pre className="channel-code" key={j}>
                  {tc.name}({JSON.stringify(tc.args)})
                </pre>
              ))}
            </div>
          </div>
        ))}

        {pendingQuery && (
          <div className="chain-row">
            <div className="chain-rail">
              <span className="chain-dot dot-hold" />
              <span className="chain-wire" />
            </div>
            <div className="channel channel-hold">
              <div className="channel-eyebrow">Hold — review required</div>
              <textarea
                className="hold-editor"
                value={editedQuery}
                onChange={(e) => setEditedQuery(e.target.value)}
                rows={4}
                spellCheck={false}
              />
              <div className="hold-actions">
                <button className="btn btn-approve" onClick={() => sendReview("approve")}>
                  Approve
                </button>
                <button
                  className="btn btn-edit"
                  onClick={() => sendReview("edit")}
                  disabled={!queryWasEdited}
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

        {finalAnswer && (
          <div className="chain-row">
            <div className="chain-rail">
              <span className="chain-dot dot-answer" />
            </div>
            <div className="channel channel-answer">
              <div className="channel-eyebrow">Answer</div>
              <p className="channel-content answer-text">{finalAnswer}</p>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
