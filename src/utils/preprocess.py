import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_BASE = Path("data/raw")
PROCESSED_BASE = Path("data/processed")

RAW_TRAIN_DIR = RAW_BASE / "train"
RAW_TEST_DIR = RAW_BASE / "test"

IMG_SIZE = (224, 224)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_SIZE = 0.2
RANDOM_STATE = 42


def preprocess_image(src_path: Path, dst_path: Path, size=(224, 224)):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(dst_path, format="JPEG", quality=95)


def collect_raw_images(split_dir: Path, split_name: str):
    rows = []

    if not split_dir.exists():
        print(f"[WARNING] Split folder not found: {split_dir}")
        return pd.DataFrame(columns=["src_path", "label", "split"])

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            rows.append({
                "src_path": str(img_path),
                "label": label,
                "split": split_name
            })

    return pd.DataFrame(rows)


def process_dataframe(df: pd.DataFrame, split_name: str):
    rows = []
    unreadable = []
    processed_count = 0

    split_output_dir = PROCESSED_BASE / split_name

    for _, row in df.iterrows():
        src_path = Path(row["src_path"])
        label = row["label"]

        out_name = src_path.stem + ".jpg"
        out_path = split_output_dir / label / out_name

        try:
            preprocess_image(src_path, out_path, size=IMG_SIZE)
            rows.append({
                "filepath": str(out_path).replace("\\", "/"),
                "label": label,
                "split": split_name
            })
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"[{split_name}] processed {processed_count} images...")

        except (UnidentifiedImageError, OSError, ValueError) as e:
            unreadable.append((str(src_path), str(e)))

    out_df = pd.DataFrame(rows)
    csv_path = PROCESSED_BASE / f"{split_name}.csv"
    out_df.to_csv(csv_path, index=False)

    print(f"\n[{split_name}] processed images: {processed_count}")
    print(f"[{split_name}] unreadable images: {len(unreadable)}")
    print(f"[{split_name}] csv saved to: {csv_path}")

    if unreadable:
        bad_df = pd.DataFrame(unreadable, columns=["filepath", "error"])
        bad_csv_path = PROCESSED_BASE / f"{split_name}_unreadable.csv"
        bad_df.to_csv(bad_csv_path, index=False)
        print(f"[{split_name}] unreadable log saved to: {bad_csv_path}")

    return out_df


def main():
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    # collect raw data
    raw_train_df = collect_raw_images(RAW_TRAIN_DIR, "train")
    raw_test_df = collect_raw_images(RAW_TEST_DIR, "test")

    if raw_train_df.empty:
        print("[ERROR] No training data found.")
        return

    # split raw train into train + validation
    train_df, validation_df = train_test_split(
        raw_train_df,
        test_size=VAL_SIZE,
        stratify=raw_train_df["label"],
        random_state=RANDOM_STATE
    )

    print("\nTrain split counts:")
    print(train_df["label"].value_counts())

    print("\nValidation split counts:")
    print(validation_df["label"].value_counts())

    # preprocess and save
    processed_train_df = process_dataframe(train_df, "train")
    processed_validation_df = process_dataframe(validation_df, "validation")

    all_dfs = [processed_train_df, processed_validation_df]

    if not raw_test_df.empty:
        print("\nTest split counts:")
        print(raw_test_df["label"].value_counts())
        processed_test_df = process_dataframe(raw_test_df, "test")
        all_dfs.append(processed_test_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = PROCESSED_BASE / "all_data.csv"
    combined_df.to_csv(combined_csv, index=False)

    print(f"\n[done] combined csv saved to: {combined_csv}")
    print("\nCombined label counts:")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()