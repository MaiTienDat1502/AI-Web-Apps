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

/*
 * Tìm ảnh bằng câu mô tả
 */
export async function searchByText(query, k = 8) {
  const formData = new FormData();

  formData.append("query", query);
  formData.append("k", String(k));

  const response = await fetch(`${API_BASE}/api/search/text`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json();
}

/*
 * Tìm ảnh bằng một ảnh mẫu
 */
export async function searchByImage(file, k = 8) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("k", String(k));

  const response = await fetch(`${API_BASE}/api/search/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json();
}

/*
 * Đường dẫn ảnh được backend lưu trong meta.json.
 *
 * Ví dụ:
 * data\images\dog.jpg
 *
 * Chuyển thành URL:
 * http://127.0.0.1:8000/images/dog.jpg
 */
export function getImageUrl(path) {
  const filename = path.replaceAll("\\", "/").split("/").pop();

  return `${API_BASE}/images/${encodeURIComponent(filename)}`;
}

export async function classifyImage(file, topK = 3) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("top_k", String(topK));

  const response = await fetch(`${API_BASE}/api/classify`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function detectObjects(file, confidence = 0.25) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("confidence", String(confidence));

  const response = await fetch(`${API_BASE}/api/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json();
}
