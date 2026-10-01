import { useState } from "react";
import { searchByText, searchByImage, getImageUrl } from "../api.js";

export default function Search() {
  const [query, setQuery] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  const [results, setResults] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [mode, setMode] = useState("text");

  async function handleTextSearch(e) {
    e.preventDefault();

    const text = query.trim();

    if (!text) {
      setError("Vui lòng nhập mô tả ảnh.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const data = await searchByText(text, 8);

      setResults(data.results || []);
    } catch (err) {
      setError(err.message || "Không thể tìm kiếm ảnh.");
    } finally {
      setLoading(false);
    }
  }

  async function handleImageSearch(e) {
    e.preventDefault();

    if (!selectedFile) {
      setError("Vui lòng chọn một ảnh.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const data = await searchByImage(selectedFile, 8);

      setResults(data.results || []);
    } catch (err) {
      setError(err.message || "Không thể tìm kiếm ảnh.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];

    setSelectedFile(file || null);
    setError("");
  }

  return (
    <section className="search">
      <h2>Image Retrieval</h2>

      <p className="muted">
        Tìm kiếm ảnh bằng câu mô tả hoặc bằng một ảnh mẫu.
      </p>

      <div className="search-tabs">
        <button
          type="button"
          className={mode === "text" ? "tab active" : "tab"}
          onClick={() => {
            setMode("text");
            setError("");
          }}
        >
          Tìm bằng văn bản
        </button>

        <button
          type="button"
          className={mode === "image" ? "tab active" : "tab"}
          onClick={() => {
            setMode("image");
            setError("");
          }}
        >
          Tìm bằng ảnh
        </button>
      </div>

      {mode === "text" && (
        <form onSubmit={handleTextSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ví dụ: a dog"
            aria-label="Mô tả ảnh"
          />

          <button type="submit" className="button" disabled={loading}>
            {loading ? "Đang tìm..." : "Tìm ảnh"}
          </button>
        </form>
      )}

      {mode === "image" && (
        <form onSubmit={handleImageSearch} className="search-form">
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleFileChange}
          />

          <button type="submit" className="button" disabled={loading}>
            {loading ? "Đang tìm..." : "Tìm ảnh tương tự"}
          </button>
        </form>
      )}

      {selectedFile && mode === "image" && (
        <p className="selected-file">
          Ảnh đã chọn: <b>{selectedFile.name}</b>
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {results.length > 0 && (
        <div className="search-results">
          <h3>Kết quả tìm kiếm ({results.length})</h3>

          <div className="image-grid">
            {results.map((item) => (
              <div className="image-card" key={item.id}>
                <img src={getImageUrl(item.path)} alt={item.label} />

                <div className="image-info">
                  <strong>{item.label}</strong>

                  <span>Score: {item.score}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
