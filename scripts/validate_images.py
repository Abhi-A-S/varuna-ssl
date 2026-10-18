from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path("raw")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    image_paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    dimensions = Counter()
    modes = Counter()
    corrupted = []
    valid = 0

    print(f"Dataset directory: {ROOT.resolve()}")
    print(f"Images found: {len(image_paths)}")

    for index, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                dimensions[image.size] += 1
                modes[image.mode] += 1

            valid += 1

        except (OSError, ValueError, UnidentifiedImageError) as error:
            corrupted.append((path, str(error)))

        if index % 10000 == 0:
            print(f"Checked {index}/{len(image_paths)}")

    print("\n--- Validation report ---")
    print(f"Valid images: {valid}")
    print(f"Corrupted/unreadable images: {len(corrupted)}")
    print(f"Dimensions: {dimensions.most_common()}")
    print(f"Image modes: {modes}")

    if corrupted:
        print("\nProblematic files:")
        for path, error in corrupted:
            print(f"{path}: {error}")

    if corrupted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
