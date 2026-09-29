const API_BASE = "http://127.0.0.1:8000";

export async function streamChat({
  message,
  history = [],
  signal,
  onSources,
  onToken,
}) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
    }),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Trình duyệt không hỗ trợ streaming response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      const line = event.split("\n").find((item) => item.startsWith("data: "));

      if (!line) continue;

      const data = JSON.parse(line.slice(6));

      if (data.type === "sources") {
        onSources?.(data.items || []);
      }

      if (data.type === "token") {
        onToken?.(data.text || "");
      }

      if (data.type === "done") {
        return;
      }
    }
  }
}
