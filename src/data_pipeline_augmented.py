from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "validation"


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    return train_ds, val_ds, class_names


def get_class_weights_from_directory(train_dir: Path, class_names: list[str]):
    counts = []
    for class_name in class_names:
        class_dir = train_dir / class_name
        count = sum(
            1 for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        counts.append(count)

    y = []
    for idx, count in enumerate(counts):
        y.extend([idx] * count)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=np.array(y),
    )

    return {i: float(w) for i, w in enumerate(weights)}


def prepare_data():
    set_seed(SEED)

    train_ds, val_ds, class_names = build_datasets()
    class_weights = get_class_weights_from_directory(TRAIN_DIR, class_names)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "class_names": class_names,
        "class_weights": class_weights,
    }