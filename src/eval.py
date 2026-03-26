# from pathlib import Path
# import json

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     f1_score,
# )
# from keras.applications.efficientnet import preprocess_input

# from src.data_pipeline_augmented import prepare_data


# MODEL_PATH = Path("experiments/results/best_efficientnetb0_augmented.keras")
# METRICS_PATH = Path("experiments/results/eval_metrics.json")
# RESULTS_CSV_PATH = Path("experiments/results/eval_results.csv")
# CONF_MATRIX_PATH = Path("experiments/results/confusion_matrix.png")
# CLASS_REPORT_PATH = Path("experiments/results/classification_report.txt")


# def main():
#     data = prepare_data()
#     val_ds = data["val_ds"]
#     class_names = data["class_names"]

#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         custom_objects={"preprocess_input": preprocess_input},
#     )

#     val_probs = model.predict(val_ds, verbose=1)
#     val_pred = np.argmax(val_probs, axis=1)

#     y_val_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

#     acc = accuracy_score(y_val_true, val_pred)
#     macro_f1 = f1_score(y_val_true, val_pred, average="macro")
#     cls_report = classification_report(y_val_true, val_pred, target_names=class_names)
#     cm = confusion_matrix(y_val_true, val_pred)

#     print("Accuracy:", round(acc, 4))
#     print("Macro-F1:", round(macro_f1, 4))
#     print("\nClassification Report:\n")
#     print(cls_report)

#     METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

#     with open(METRICS_PATH, "w") as f:
#         json.dump(
#             {
#                 "accuracy": float(acc),
#                 "macro_f1": float(macro_f1),
#             },
#             f,
#             indent=2,
#         )

#     with open(CLASS_REPORT_PATH, "w") as f:
#         f.write(cls_report)

#     results_df = pd.DataFrame({
#         "Model": ["EfficientNetB0"],
#         "Accuracy": [acc],
#         "Macro-F1": [macro_f1],
#     })
#     results_df.to_csv(RESULTS_CSV_PATH, index=False)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
#     plt.title("EfficientNetB0 Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches="tight")
#     plt.show()

#     print(f"Saved metrics to {METRICS_PATH}")
#     print(f"Saved results table to {RESULTS_CSV_PATH}")
#     print(f"Saved confusion matrix to {CONF_MATRIX_PATH}")
#     print(f"Saved classification report to {CLASS_REPORT_PATH}")


# if __name__ == "__main__":
#     main()


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
from src.utils.grad_cam import generate_and_save_gradcam


MODEL_PATH = Path("experiments/results/best_efficientnetb0_augmented.keras")
METRICS_PATH = Path("experiments/results/eval_metrics.json")
RESULTS_CSV_PATH = Path("experiments/results/eval_results.csv")
CONF_MATRIX_PATH = Path("experiments/results/confusion_matrix.png")
CLASS_REPORT_PATH = Path("experiments/results/classification_report.txt")
GRADCAM_DIR = Path("experiments/results/gradcam_eval")

LAST_CONV_LAYER_NAME = "top_conv"
NUM_CORRECT_GRADCAM = 3
NUM_WRONG_GRADCAM = 3
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_validation_image_paths(class_names):
    image_paths = []
    true_labels = []

    val_root = Path("data/processed/validation")

    for class_index, class_name in enumerate(class_names):
        class_dir = val_root / class_name
        if not class_dir.exists():
            continue

        class_files = sorted(
            [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
            ]
        )

        for img_path in class_files:
            image_paths.append(img_path)
            true_labels.append(class_index)

    return image_paths, np.array(true_labels)


def pick_gradcam_indices(y_true, y_pred):
    correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    wrong_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

    selected_correct = correct_indices[:NUM_CORRECT_GRADCAM]
    selected_wrong = wrong_indices[:NUM_WRONG_GRADCAM]

    return selected_correct + selected_wrong


def generate_gradcam_examples(model, image_paths, y_true, y_pred, y_probs, class_names):
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    selected_indices = pick_gradcam_indices(y_true, y_pred)

    if not selected_indices:
        print("No Grad-CAM samples selected.")
        return

    for idx in selected_indices:
        img_path = image_paths[idx]
        true_idx = int(y_true[idx])
        pred_idx = int(y_pred[idx])
        confidence = float(y_probs[idx][pred_idx])

        true_label = class_names[true_idx]
        predicted_label = class_names[pred_idx]

        tag = "correct" if true_idx == pred_idx else "wrong"
        output_name = f"{tag}_{idx:03d}_true-{true_label}_pred-{predicted_label}.png"
        output_path = GRADCAM_DIR / output_name

        generate_and_save_gradcam(
            img_path=img_path,
            model=model,
            output_path=output_path,
            last_conv_layer_name=LAST_CONV_LAYER_NAME,
            pred_index=pred_idx,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
            alpha=0.4,
        )

        print(f"Saved Grad-CAM: {output_path}")


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

    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("\nClassification Report:\n")
    print(cls_report)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
            },
            f,
            indent=2,
        )

    with open(CLASS_REPORT_PATH, "w") as f:
        f.write(cls_report)

    results_df = pd.DataFrame(
        {
            "Model": ["EfficientNetB0"],
            "Accuracy": [acc],
            "Macro-F1": [macro_f1],
        }
    )
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("EfficientNetB0 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    image_paths, y_from_paths = collect_validation_image_paths(class_names)

    if len(image_paths) != len(y_val_true):
        print("Warning: validation image count does not match validation label count.")
        print("Skipping Grad-CAM generation.")
    elif not np.array_equal(y_from_paths, y_val_true):
        print("Warning: validation file order does not match dataset label order.")
        print("Skipping Grad-CAM generation.")
    else:
        print("\nGenerating Grad-CAM samples...\n")
        generate_gradcam_examples(
            model=model,
            image_paths=image_paths,
            y_true=y_val_true,
            y_pred=val_pred,
            y_probs=val_probs,
            class_names=class_names,
        )

    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved results table to {RESULTS_CSV_PATH}")
    print(f"Saved confusion matrix to {CONF_MATRIX_PATH}")
    print(f"Saved classification report to {CLASS_REPORT_PATH}")
    print(f"Saved Grad-CAM outputs to {GRADCAM_DIR}")


if __name__ == "__main__":
    main()