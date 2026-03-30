from pathlib import Path
import json

import tensorflow as tf
from keras import layers, models, regularizers
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

from src.data_pipeline_augmented import prepare_data

MODEL_OUTPUT_PATH = Path("experiments/results/ablations/best_efficientnetb0_no_augmentation.keras")
HISTORY_OUTPUT_PATH = Path("experiments/logs/ablations/training_history_no_augmentation.json")
CLASS_NAMES_OUTPUT_PATH = Path("experiments/logs/ablations/class_names_no_augmentation.json")


def build_efficientnet_no_augmentation(num_classes: int):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Lambda(preprocess_input)(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def get_callbacks(model_path: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_history(history_dict: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history_dict, f, indent=2)


def save_class_names(class_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(class_names), f, indent=2)


def main():
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    class_names = data["class_names"]
    class_weights = data["class_weights"]
    num_classes = len(class_names)

    print("Experiment: No Augmentation")
    print("Class names:", class_names)
    print("Class weights:", class_weights)

    save_class_names(class_names, CLASS_NAMES_OUTPUT_PATH)
    print(f"Class names saved to {CLASS_NAMES_OUTPUT_PATH}")

    model, base_model = build_efficientnet_no_augmentation(num_classes)

    print("\nStage 1: training head...\n")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=get_callbacks(str(MODEL_OUTPUT_PATH)),
        verbose=1,
    )

    print("\nStage 2: fine-tuning top layers...\n")
    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=get_callbacks(str(MODEL_OUTPUT_PATH)),
        verbose=1,
    )

    combined_history = {}
    all_keys = set(history1.history.keys()).union(set(history2.history.keys()))
    for key in all_keys:
        combined_history[key] = history1.history.get(key, []) + history2.history.get(key, [])

    combined_history["ablation"] = "no_augmentation"

    save_history(combined_history, HISTORY_OUTPUT_PATH)

    print(f"\nTraining complete. Best model saved to {MODEL_OUTPUT_PATH}")
    print(f"Training history saved to {HISTORY_OUTPUT_PATH}")
    print(f"Class names saved to {CLASS_NAMES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()