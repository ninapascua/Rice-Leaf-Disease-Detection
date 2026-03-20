import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras.applications.efficientnet import preprocess_input


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/validation.csv")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataframes(
    train_csv: Path = TRAIN_CSV,
    val_csv: Path = VAL_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_df["filepath"] = train_df["filepath"].str.replace(
        "data/processed/train", "data/raw/RiceLeafs/train", regex=False
    )
    val_df["filepath"] = val_df["filepath"].str.replace(
        "data/processed/validation", "data/raw/RiceLeafs/validation", regex=False
    )

    train_df["full_path"] = train_df["filepath"].apply(lambda p: os.path.join(".", p))
    val_df["full_path"] = val_df["filepath"].apply(lambda p: os.path.join(".", p))

    train_df = train_df[train_df["full_path"].apply(os.path.exists)].reset_index(drop=True)
    val_df = val_df[val_df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    return train_df, val_df


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    class_names = label_encoder.classes_
    return y_train, y_val, label_encoder, class_names


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    return {i: float(w) for i, w in enumerate(class_weights_array)}


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTED pipeline — replaces the 2-op baseline
# Strategy: rice leaf diseases are distinguished by colour/texture patterns,
# so we augment spatial geometry, colour, and fine texture independently.
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img, label):
    # ── 1. Spatial / geometric ────────────────────────────────────────────────
    # Flip both axes: disease patches appear anywhere on the leaf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Random 90° rotation (k ∈ {0,1,2,3}) — leaves have no canonical orientation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    # Random crop → resize: simulates zoom-in / scale variation
    # Crop to 85–100 % of area, then resize back to IMG_SIZE
    crop_frac = tf.random.uniform([], minval=0.85, maxval=1.0)
    crop_h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * crop_frac, tf.int32)
    crop_w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * crop_frac, tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMG_SIZE)

    # ── 2. Colour / photometric ───────────────────────────────────────────────
    # Stronger brightness swing than baseline (±0.08 → ±0.15)
    img = tf.image.random_brightness(img, max_delta=0.2)

    # Contrast jitter: compresses/expands dynamic range
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    # Saturation jitter: simulates lighting condition / camera variation
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)

    # Hue jitter (small): leaves can have slight tonal variation across images
    img = tf.image.random_hue(img, max_delta=0.03)

    # ── 3. Fine-grained noise ─────────────────────────────────────────────────
    # Gaussian noise: makes model robust to camera sensor noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=5.0)
    img = img + noise

    # ── 4. Clip to valid EfficientNet preprocess_input range ─────────────────
    # preprocess_input maps [0,255] → roughly [-1, 1]; clip keeps values sane
    img = tf.clip_by_value(img, -128.0, 128.0)

    return img, label


def make_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_paths = train_df["full_path"].values
    val_paths = val_df["full_path"].values

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, y_train))
        .shuffle(len(train_paths), seed=42, reshuffle_each_iteration=True)
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .map(augment_image, num_parallel_calls=AUTOTUNE)   # ← augmented
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Validation: NO augmentation — deterministic eval
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, y_val))
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds


def prepare_data():
    set_seed(42)
    train_df, val_df = load_dataframes()
    y_train, y_val, label_encoder, class_names = encode_labels(train_df, val_df)
    class_weights = get_class_weights(y_train)
    train_ds, val_ds = make_datasets(train_df, val_df, y_train, y_val)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "y_train": y_train,
        "y_val": y_val,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "class_weights": class_weights,
        "train_ds": train_ds,
        "val_ds": val_ds,
    }
