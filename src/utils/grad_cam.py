from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img


IMG_SIZE = (224, 224)


def get_img_array(img_path, size=IMG_SIZE):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0).astype("float32")
    return array


def find_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("EfficientNet base model not found inside the loaded model.")


def safe_call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def build_prefix_model(model, base_model):
    base_index = model.layers.index(base_model)
    prefix_input = tf.keras.Input(shape=(224, 224, 3))
    x = prefix_input

    for layer in model.layers[1:base_index]:
        x = safe_call_layer(layer, x)

    return tf.keras.Model(prefix_input, x, name="gradcam_prefix_model")


def build_classifier_model(model, base_model, last_conv_layer_name="top_conv"):
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_index = base_model.layers.index(last_conv_layer)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    for layer in base_model.layers[last_conv_index + 1:]:
        x = safe_call_layer(layer, x)

    base_index = model.layers.index(base_model)
    for layer in model.layers[base_index + 1:]:
        x = safe_call_layer(layer, x)

    return tf.keras.Model(classifier_input, x, name="gradcam_classifier_model")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    base_model = find_base_model(model)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    prefix_model = build_prefix_model(model, base_model)

    last_conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output,
        name="gradcam_last_conv_model",
    )

    classifier_model = build_classifier_model(model, base_model, last_conv_layer_name)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    preprocessed_input = prefix_model(img_tensor, training=False)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_model(preprocessed_input, training=False)
        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs, training=False)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val.numpy()) == 0.0:
        return heatmap.numpy()

    heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    img = load_img(img_path)
    img = img_to_array(img).astype("uint8")

    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colorized_heatmap = colormap(np.arange(256))[:, :3]
    colorized_heatmap = colorized_heatmap[heatmap_uint8]
    colorized_heatmap = tf.keras.utils.array_to_img(colorized_heatmap)
    colorized_heatmap = colorized_heatmap.resize((img.shape[1], img.shape[0]))
    colorized_heatmap = img_to_array(colorized_heatmap)

    superimposed_img = colorized_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return img, superimposed_img


def save_gradcam_figure(
    img_path,
    heatmap,
    output_path,
    true_label=None,
    predicted_label=None,
    confidence=None,
    alpha=0.6,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_img, gradcam_img = overlay_heatmap_on_image(img_path, heatmap, alpha=alpha)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gradcam_img)

    title_parts = ["Grad-CAM"]
    if true_label is not None:
        title_parts.append(f"True: {true_label}")
    if predicted_label is not None:
        title_parts.append(f"Pred: {predicted_label}")
    if confidence is not None:
        title_parts.append(f"Conf: {confidence:.2%}")

    plt.title("\n".join(title_parts), fontsize=10)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_and_save_gradcam(
    img_path,
    model,
    output_path,
    last_conv_layer_name="top_conv",
    pred_index=None,
    true_label=None,
    predicted_label=None,
    confidence=None,
    alpha=0.4,
):
    img_array = get_img_array(img_path)
    heatmap = make_gradcam_heatmap(
        img_array=img_array,
        model=model,
        last_conv_layer_name=last_conv_layer_name,
        pred_index=pred_index,
    )

    save_gradcam_figure(
        img_path=img_path,
        heatmap=heatmap,
        output_path=output_path,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        alpha=alpha,
    )