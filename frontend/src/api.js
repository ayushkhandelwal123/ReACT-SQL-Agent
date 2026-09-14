/**
 * api.js
 * ------
 * Talks to the FastAPI backend's streaming endpoints.
 *
 * The browser's built-in EventSource only supports GET requests, and we
 * need to POST a question (arbitrary length) — so instead we use fetch()
 * and read the response body as a stream ourselves, parsing the same
 * "data: {...}\n\n" SSE wire format by hand. This is a completely normal
 * pattern for POST-based SSE; the ONLY thing we lose vs EventSource is
 * automatic reconnection, which we don't need for a single request/response
 * agent run.
 */

const API_BASE = "http://127.0.0.1:8000";

/**
 * POST to `path` with `body`, stream the SSE response, and call
 * `onEvent(data)` for each parsed JSON event as it arrives.
 */
export async function streamRequest(path, body, onEvent) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    let detail = `Request failed (${response.status})`;
    try {
      const errJson = await response.json();
      detail = errJson.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line. The last chunk in the
    // buffer might be a partial event still arriving, so keep it around
    // for the next read instead of parsing it early.
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const raw of events) {
      const line = raw.trim();
      if (!line.startsWith("data:")) continue;
      const jsonStr = line.slice("data:".length).trim();
      if (!jsonStr) continue;
      try {
        onEvent(JSON.parse(jsonStr));
      } catch (err) {
        console.error("Could not parse SSE event:", jsonStr, err);
      }
    }
  }
}

export function askStream(question, threadId, onEvent) {
  // No thread_id on the very first question — the backend creates one and
  // hands it back in every event. Every question after that passes it
  // along, which is what makes this an actual multi-turn conversation
  // instead of N unrelated single questions: the checkpointer on the
  // backend keeps appending to the SAME message history for that thread_id.
  const body = threadId ? { question, thread_id: threadId } : { question };
  return streamRequest("/ask/stream", body, onEvent);
}

export function reviewStream(threadId, decision, onEvent) {
  return streamRequest(`/review/${threadId}/stream`, decision, onEvent);
}
