from pathlib import Path

from PIL import Image

from core.detector import ObjectDetector


ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "data" / "images" / "dog.jpg"


def main():
    print("Đang load YOLO11n...")

    detector = ObjectDetector()

    print("Đang nhận diện:")
    print(IMAGE_PATH)

    image = Image.open(IMAGE_PATH)

    result = detector.predict(image)

    print()
    print("================================")
    print("OBJECT DETECTION RESULT")
    print("================================")

    print(f"Số object: {result['count']}")

    for item in result["detections"]:
        print(
            f"- {item['label']}: "
            f"{item['score'] * 100:.2f}% "
            f"box={item['box']}"
        )


if __name__ == "__main__":
    main()