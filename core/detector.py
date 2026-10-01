"""Object Detection bằng YOLO11n."""

from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from config import ART_DIR


class ObjectDetector:
    def __init__(
        self,
        model_path: Path = ART_DIR / "detector" / "yolo11n.pt",
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy YOLO model: {model_path}"
            )

        print(f"Loading YOLO model: {model_path}")

        self.model = YOLO(str(model_path))

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.25,
    ) -> dict:
        """
        Phát hiện object trong ảnh.

        Trả về:
        - count
        - detections
        - annotated_image
        """

        image = image.convert("RGB")

        results = self.model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                score = float(
                    boxes.conf[i].item()
                )

                xyxy = boxes.xyxy[i].tolist()

                label = self.model.names[class_id]

                detections.append(
                    {
                        "label": label,
                        "class_id": class_id,
                        "score": round(score, 4),
                        "box": [
                            round(float(x), 2)
                            for x in xyxy
                        ],
                    }
                )

        # YOLO tự vẽ bounding box
        annotated = result.plot()

        # Ultralytics trả ảnh dạng BGR.
        # Chuyển sang RGB để PIL xử lý đúng.
        annotated = Image.fromarray(
            annotated[:, :, ::-1]
        )

        return {
            "count": len(detections),
            "detections": detections,
            "annotated_image": annotated,
        }