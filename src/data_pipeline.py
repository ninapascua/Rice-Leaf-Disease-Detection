import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input


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
        "data/processed/train", "data/raw/train", regex=False
    )
    val_df["filepath"] = val_df["filepath"].str.replace(
        "data/processed/validation", "data/raw/validation", regex=False
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


def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
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
        .map(augment_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

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