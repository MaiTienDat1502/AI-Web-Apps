import { useState } from "react";
import { detectObjects } from "../api.js";

export default function Detect() {
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
      setResult(null);
      setError("");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  async function handleDetect() {
    if (!file || busy) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const data = await detectObjects(file, 0.5);

      setResult(data);
    } catch (err) {
      setError(err.message || "Không thể thực hiện Object Detection.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="feature-card">
      <h2>Object Detection</h2>

      <p className="muted">
        Phát hiện các đối tượng trong ảnh bằng mô hình YOLO11n.
      </p>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {/* Ảnh gốc */}
      {preview && !result && (
        <div className="classify-preview">
          <h3>Ảnh gốc</h3>

          <img src={preview} alt="Ảnh được chọn" />
        </div>
      )}

      <button
        type="button"
        className="button"
        onClick={handleDetect}
        disabled={!file || busy}
      >
        {busy ? "Đang nhận diện..." : "Nhận diện object"}
      </button>

      {error && <p className="error">Lỗi: {error}</p>}

      {result && (
        <div className="classification-result">
          <h3>Kết quả Object Detection</h3>

          {/* Ảnh đã vẽ bounding box */}
          {result.image && (
            <div className="detection-result-image">
              <h4>Ảnh sau khi nhận diện</h4>

              <img src={result.image} alt="Ảnh với bounding box" />
            </div>
          )}

          <p>
            Phát hiện được <strong>{result.count}</strong> object.
          </p>

          {result.detections.length === 0 ? (
            <p className="muted">Không phát hiện được object nào.</p>
          ) : (
            <>
              <h4>Danh sách object</h4>

              {result.detections.map((item, index) => (
                <div className="prediction" key={index}>
                  <span>{item.label}</span>

                  <strong>{(item.score * 100).toFixed(2)}%</strong>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </section>
  );
}
