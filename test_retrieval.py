from PIL import Image

from core.retrieval import ImageSearch


def main():
    print("Đang tải ImageSearch...")
    searcher = ImageSearch()

    # Đổi tên file này thành một ảnh thực tế trong data/images
    query_path = "data/images/dog.jpg"

    print(f"\nẢnh truy vấn: {query_path}")

    image = Image.open(query_path)

    results = searcher.search_image(
        image,
        k=5
    )

    print("\nKết quả:")
    for i, result in enumerate(results, 1):
        print(
            f"{i}. "
            f"{result['label']} | "
            f"score={result['score']} | "
            f"path={result['path']}"
        )


if __name__ == "__main__":
    main()