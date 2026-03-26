from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = Path("experiments/results/best_efficientnetb0_augmented.keras")
CLASS_NAMES_PATH = Path("experiments/logs/class_names.json")
IMG_SIZE = (224, 224)


def load_class_names(class_names_path=CLASS_NAMES_PATH):
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    with open(class_names_path, "r") as f:
        return json.load(f)


def pretty_class_name(name: str) -> str:
    mapping = {
        "BrownSpot": "BrownSpot",
        "Healthy": "Healthy",
        "leaf_blast": "LeafBlast",
        "rice_hispa": "Hispa",
        "LeafBlast": "LeafBlast",
        "Hispa": "Hispa",
    }
    return mapping.get(name, name)


def load_trained_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return tf.keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )


def preprocess_image_file(image_path):
    img = tf.io.read_file(str(image_path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def predict_image_file(model, image_path):
    raw_class_names = load_class_names()
    display_class_names = [pretty_class_name(name) for name in raw_class_names]

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    img = preprocess_image_file(image_path)
    preds = model.predict(img, verbose=0)[0]

    pred_idx = int(tf.argmax(preds).numpy())
    predicted_class = display_class_names[pred_idx]
    confidence = float(preds[pred_idx])

    probabilities = {
        display_name: float(prob)
        for display_name, prob in zip(display_class_names, preds)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }