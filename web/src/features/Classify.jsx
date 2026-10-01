import { useState } from "react";
import { classifyImage } from "../api.js";

export default function Classify() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(e) {
    const selectedFile = e.target.files?.[0];

    if (!selectedFile) {
      setFile(null);
      setPreview("");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  async function handleClassify() {
    if (!file || busy) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const data = await classifyImage(file, 3);
      setResult(data);
    } catch (err) {
      setError(err.message || "Không thể phân loại ảnh.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="feature-card">
      <h2>Image Classification</h2>

      <p className="muted">Phân loại ảnh hoa bằng mô hình ResNet-18.</p>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {preview && (
        <div className="classify-preview">
          <img src={preview} alt="Ảnh được chọn" />
        </div>
      )}

      <button
        type="button"
        className="button"
        onClick={handleClassify}
        disabled={!file || busy}
      >
        {busy ? "Đang phân loại..." : "Phân loại ảnh"}
      </button>

      {error && <p className="error">Lỗi: {error}</p>}

      {result && (
        <div className="classification-result">
          <h3>Kết quả</h3>

          {result.predictions.map((item, index) => (
            <div className="prediction" key={index}>
              <span>{item.label}</span>

              <strong>{(item.score * 100).toFixed(2)}%</strong>
            </div>
          ))}

          <p className="confidence">
            {result.confident
              ? "✓ Độ tin cậy đạt yêu cầu"
              : "⚠ Độ tin cậy chưa cao"}
          </p>
        </div>
      )}
    </section>
  );
}
