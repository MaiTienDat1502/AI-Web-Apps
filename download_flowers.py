from pathlib import Path
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FLOWERS_DIR = DATA_DIR / "flowers" / "flower_photos"

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE = DATA_DIR / "flower_photos.tgz"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if FLOWERS_DIR.exists():
        print("Dataset đã tồn tại:")
        print(FLOWERS_DIR)
        return

    print("Đang tải TF Flowers...")
    print("Dung lượng tải xuống có thể khá lớn, hãy chờ.")

    urllib.request.urlretrieve(
        URL,
        ARCHIVE,
    )

    print("Đã tải xong.")
    print("Đang giải nén...")

    FLOWERS_ROOT = DATA_DIR / "flowers"
    FLOWERS_ROOT.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(
            FLOWERS_ROOT,
            filter="data",
        )

    ARCHIVE.unlink(missing_ok=True)

    if (FLOWERS_DIR / "LICENSE.txt").exists():
        (FLOWERS_DIR / "LICENSE.txt").unlink()

    print()
    print("Đã chuẩn bị dataset.")

    for folder in sorted(FLOWERS_DIR.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg")))
            print(f"{folder.name}: {count} ảnh")


if __name__ == "__main__":
    main()