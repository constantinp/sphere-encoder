import argparse
import json
import random
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Sphere embedding dataset manifest from Aphrodite-style caption sidecars."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing images and {image.ext}_cap.json sidecars.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save train.json and val.json.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed for the train/val split.",
    )
    return parser.parse_args()


def caption_sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + "_cap.json")


def iter_dataset_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def has_usable_caption_sidecar(image_path: Path) -> bool:
    sidecar_path = caption_sidecar_path(image_path)
    if not sidecar_path.exists():
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    return any(isinstance(value, str) and value.strip() for value in payload.values())


def write_manifest(path: Path, entries: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    print(f"Scanning {img_dir}...")
    for image_path in iter_dataset_images(img_dir):
        if not has_usable_caption_sidecar(image_path):
            continue
        dataset.append(
            {
                "image_path": str(image_path.absolute()),
                "is_absolute_path": True,
            }
        )

    if not dataset:
        print("No images with usable _cap.json sidecars were found.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    val_size = int(len(dataset) * args.val_split)
    val_data = dataset[:val_size]
    train_data = dataset[val_size:]

    write_manifest(output_dir / "train.json", train_data)
    write_manifest(output_dir / "val.json", val_data)

    print(f"Done! Processed {len(dataset)} images with caption sidecars.")
    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")


if __name__ == "__main__":
    main()
