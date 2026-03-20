from pathlib import Path
import json

import tensorflow as tf

from src.data_pipeline import prepare_data
from src.models.cnn_model import build_efficientnet


MODEL_OUTPUT_PATH = Path("experiments/results/best_efficientnetb0.keras")
HISTORY_OUTPUT_PATH = Path("experiments/logs/training_history.json")


def get_callbacks(model_path: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_history(history_dict: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history_dict, f, indent=2)


def combine_histories(history1, history2):
    combined_history = {}

    all_keys = set(history1.history.keys()) | set(history2.history.keys())
    for key in all_keys:
        combined_history[key] = history1.history.get(key, []) + history2.history.get(key, [])

    return combined_history


def main():
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    class_weights = data["class_weights"]
    num_classes = len(data["class_names"])

    print("Class weights:", class_weights)

    model, base_model = build_efficientnet(num_classes)
    callbacks = get_callbacks(str(MODEL_OUTPUT_PATH))

    print("Starting first training phase...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("Starting fine-tuning phase...")
    base_model.trainable = True
    for layer in base_model.layers[:-150]:
        layer.trainable = False


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        initial_epoch=history.epoch[-1] + 1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = combine_histories(history, fine_tune_history)
    save_history(combined_history, HISTORY_OUTPUT_PATH)

    print(f"Training complete. Model saved to {MODEL_OUTPUT_PATH}")
    print(f"Training history saved to {HISTORY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()