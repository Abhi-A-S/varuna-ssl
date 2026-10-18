import csv
from collections import Counter
from pathlib import Path

DATASET_DIR = Path("raw/airbus-ship-detection")
IMAGE_DIR = DATASET_DIR / "train_v2"
ANNOTATION_FILE = DATASET_DIR / "train_ship_segmentations_v2.csv"


def main():
    if not ANNOTATION_FILE.is_file():
        raise SystemExit(f"Annotation CSV not found: {ANNOTATION_FILE}")

    rows = 0
    empty_rows = 0
    annotated_rows = 0
    image_rows = Counter()

    with ANNOTATION_FILE.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows += 1
            image_id = row["ImageId"]
            encoded_pixels = (row["EncodedPixels"] or "").strip()

            image_rows[image_id] += 1

            if encoded_pixels:
                annotated_rows += 1
            else:
                empty_rows += 1

    all_images = set(image_rows)
    ship_images = {
        image_id for image_id, count in image_rows.items() if count > 0
    }

    # Re-read the CSV to identify images with at least one ship mask.
    ship_images.clear()

    with ANNOTATION_FILE.open(newline="") as file:
        for row in csv.DictReader(file):
            if (row["EncodedPixels"] or "").strip():
                ship_images.add(row["ImageId"])

    empty_images = all_images - ship_images
    image_files = (
        {path.name for path in IMAGE_DIR.glob("*") if path.is_file()}
        if IMAGE_DIR.is_dir()
        else set()
    )

    print("=== Airbus Dataset Analysis ===")
    print(f"Annotation rows: {rows:,}")
    print(f"Rows with masks: {annotated_rows:,}")
    print(f"Rows without masks: {empty_rows:,}")
    print(f"Unique annotated image IDs: {len(all_images):,}")
    print(f"Images with at least one ship: {len(ship_images):,}")
    print(f"Images without ships: {len(empty_images):,}")
    print(f"Training image files found: {len(image_files):,}")

    missing = all_images - image_files
    unlisted = image_files - all_images

    print(f"Annotation IDs missing from image directory: {len(missing):,}")
    print(f"Images absent from annotation CSV: {len(unlisted):,}")

    if missing:
        print("First missing image IDs:", sorted(missing)[:10])


if __name__ == "__main__":
    main()
