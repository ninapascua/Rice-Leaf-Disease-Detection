# from src.utils.inference import load_trained_model, predict_image_file


# def main():
#     model = load_trained_model()

#     print("Rice Leaf Disease Detection - Image Predictor")
#     print("Type the full image path to test one image.")
#     print("Type 'exit' to quit.")

#     while True:
#         image_path = input("\nEnter image path: ").strip()

#         if image_path.lower() == "exit":
#             print("Exiting predictor.")
#             break

#         if not image_path:
#             print("Please enter a valid image path.")
#             continue

#         try:
#             result = predict_image_file(model, image_path)

#             print("\nPrediction Result")
#             print(f"Predicted Class: {result['predicted_class']}")
#             print(f"Confidence: {result['confidence']:.2%}")

#             print("\nClass Probabilities:")
#             for class_name, prob in result["probabilities"].items():
#                 print(f"{class_name}: {prob:.2%}")

#         except FileNotFoundError:
#             print("File not found. Check the image path and try again.")
#         except Exception as e:
#             print(f"Error: {e}")


# if __name__ == "__main__":
#     main()

from pathlib import Path

from src.utils.grad_cam import generate_and_save_gradcam
from src.utils.inference import load_trained_model, predict_image_file


GRADCAM_PREDICT_DIR = Path("experiments/results/gradcam_predict")
LAST_CONV_LAYER_NAME = "top_conv"


def main():
    model = load_trained_model()
    GRADCAM_PREDICT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rice Leaf Disease Detection - Image Predictor")
    print("Type the full image path to test one image.")
    print("Type 'exit' to quit.")

    while True:
        image_path = input("\nEnter image path: ").strip()

        if image_path.lower() == "exit":
            print("Exiting predictor.")
            break

        if not image_path:
            print("Please enter a valid image path.")
            continue

        try:
            result = predict_image_file(model, image_path)

            print("\nPrediction Result")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")

            print("\nClass Probabilities:")
            for class_name, prob in result["probabilities"].items():
                print(f"{class_name}: {prob:.2%}")

            image_path_obj = Path(image_path)
            output_name = f"{image_path_obj.stem}_gradcam.png"
            output_path = GRADCAM_PREDICT_DIR / output_name

            generate_and_save_gradcam(
                img_path=image_path_obj,
                model=model,
                output_path=output_path,
                last_conv_layer_name=LAST_CONV_LAYER_NAME,
                pred_index=result["predicted_index"],
                true_label=None,
                predicted_label=result["predicted_class"],
                confidence=result["confidence"],
                alpha=0.4,
            )

            print(f"\nGrad-CAM saved to: {output_path}")

        except FileNotFoundError:
            print("File not found. Check the image path and try again.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()