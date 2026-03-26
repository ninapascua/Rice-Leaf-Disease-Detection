from src.utils.inference import load_trained_model, predict_image_file


def main():
    model = load_trained_model()

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

        except FileNotFoundError:
            print("File not found. Check the image path and try again.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()