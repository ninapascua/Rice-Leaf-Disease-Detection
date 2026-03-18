import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd

RAW_BASE = Path("data/raw/RiceLeafs")
PROCESSED_BASE = Path("data/processed")

SPLITS = ["train", "validation"]
IMG_SIZE = (224, 224)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def preprocess_image(src_path: Path, dst_path: Path, size=(224, 224)):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(dst_path, format="JPEG", quality=95)

def process_split(split: str):
    split_input_dir = RAW_BASE / split
    split_output_dir = PROCESSED_BASE / split

    rows = []
    unreadable = []
    processed_count = 0

    if not split_input_dir.exists():
        print(f"[WARNING] Split folder not found: {split_input_dir}")
        return None

    for class_dir in sorted(split_input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        print(f"\nProcessing class: {label}")

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            out_name = img_path.stem + ".jpg"
            out_path = split_output_dir / label / out_name

            try:
                preprocess_image(img_path, out_path, size=IMG_SIZE)
                rows.append({
                    "filepath": str(out_path).replace("\\", "/"),
                    "label": label,
                    "split": split
                })
                processed_count += 1

                if processed_count % 50 == 0:
                    print(f"[{split}] processed {processed_count} images...")

            except (UnidentifiedImageError, OSError, ValueError) as e:
                unreadable.append((str(img_path), str(e)))

    df = pd.DataFrame(rows)
    csv_path = PROCESSED_BASE / f"{split}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[{split}] processed images: {processed_count}")
    print(f"[{split}] unreadable images: {len(unreadable)}")
    print(f"[{split}] csv saved to: {csv_path}")

    if unreadable:
        bad_df = pd.DataFrame(unreadable, columns=["filepath", "error"])
        bad_csv_path = PROCESSED_BASE / f"{split}_unreadable.csv"
        bad_df.to_csv(bad_csv_path, index=False)
        print(f"[{split}] unreadable log saved to: {bad_csv_path}")

    return df

def main():
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for split in SPLITS:
        print(f"\nStarting split: {split}")
        df = process_split(split)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = PROCESSED_BASE / "all_data.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n[done] combined csv saved to: {combined_csv}")
        print(combined_df["label"].value_counts())
    else:
        print("[ERROR] No data processed.")

if __name__ == "__main__":
    main()