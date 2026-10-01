from pathlib import Path

from PIL import Image

from core.classifier import ImageClassifier


ROOT = Path(__file__).resolve().parent

FLOWERS_DIR = (
    ROOT
    / "data"
    / "flowers"
    / "flower_photos"
)


def main():

    classifier = ImageClassifier()

    test_image = next(
        (FLOWERS_DIR / "sunflowers").glob("*.jpg")
    )

    print("Ảnh test:")
    print(test_image)
    print()

    result = classifier.predict(
        Image.open(test_image),
        top_k=3,
    )

    print("Kết quả:")

    for item in result["predictions"]:

        print(
            f"{item['label']}: "
            f"{item['score'] * 100:.2f}%"
        )

    print()

    print(
        "Confident:",
        result["confident"],
    )


if __name__ == "__main__":
    main()