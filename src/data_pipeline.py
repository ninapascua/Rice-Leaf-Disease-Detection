from pathlib import Path
import pandas as pd

def load_split_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)

def explain_prediction(label):
    explanations = {
        "BrownSpot": "Brown spot appears as small brown lesions on rice leaves and may reduce yield.",
        "LeafBlast": "Leaf blast causes spindle-shaped lesions and is a major rice disease.",
        "Hispa": "Hispa damage appears as white streaks caused by insect feeding.",
        "Healthy": "The leaf shows no visible signs of disease."
    }
    return explanations.get(label, "No explanation available.")

if __name__ == "__main__":
    print(explain_prediction("BrownSpot"))