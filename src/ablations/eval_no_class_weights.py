from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data_pipeline_augmented import prepare_data


MODEL_PATH = Path("experiments/results/ablations/best_efficientnetb0_no_class_weights.keras")
METRICS_PATH = Path("experiments/results/ablations/eval_metrics_no_class_weights.json")
RESULTS_CSV_PATH = Path("experiments/results/ablations/eval_results_no_class_weights.csv")
CONF_MATRIX_PATH = Path("experiments/results/ablations/confusion_matrix_no_class_weights.png")
CLASS_REPORT_PATH = Path("experiments/results/ablations/classification_report_no_class_weights.txt")


def main():
    data = prepare_data()
    val_ds = data["val_ds"]
    class_names = data["class_names"]

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

    val_probs = model.predict(val_ds, verbose=1)
    val_pred = np.argmax(val_probs, axis=1)
    y_val_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

    acc = accuracy_score(y_val_true, val_pred)
    macro_f1 = f1_score(y_val_true, val_pred, average="macro")
    cls_report = classification_report(y_val_true, val_pred, target_names=class_names)
    cm = confusion_matrix(y_val_true, val_pred)

    print("Experiment: No Class Weights")
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("\nClassification Report:\n")
    print(cls_report)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "experiment": "no_class_weights",
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
            },
            f,
            indent=2,
        )

    with open(CLASS_REPORT_PATH, "w") as f:
        f.write(cls_report)

    results_df = pd.DataFrame({
        "Model": ["EfficientNetB0 - No Class Weights"],
        "Accuracy": [acc],
        "Macro-F1": [macro_f1],
    })
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("EfficientNetB0 Confusion Matrix - No Class Weights")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved results table to {RESULTS_CSV_PATH}")
    print(f"Saved confusion matrix to {CONF_MATRIX_PATH}")
    print(f"Saved classification report to {CLASS_REPORT_PATH}")


if __name__ == "__main__":
    main()