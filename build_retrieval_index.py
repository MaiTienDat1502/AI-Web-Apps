from pathlib import Path

from core.retrieval import ClipEncoder, build_index


IMAGE_DIR = Path("data/images")


def main():
    image_files = []

    for path in IMAGE_DIR.iterdir():
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            image_files.append(path)

    image_files.sort()

    if not image_files:
        print("Không tìm thấy ảnh trong data/images")
        return

    items = []

    for path in image_files:
        items.append({
            "path": str(path),
            "label": path.stem,
            "source": "custom",
        })

    print(f"Tìm thấy {len(items)} ảnh.")

    print("Đang tải CLIP...")
    encoder = ClipEncoder()

    print("Đang tạo FAISS index...")
    build_index(
        encoder,
        items,
    )

    print("Đã tạo xong FAISS index.")
    print("Vị trí: artifacts/retrieval/")


if __name__ == "__main__":
    main()