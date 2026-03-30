# Rice Leaf Disease Detection using CNN, NLP, Reinforcement Learning, and Explainability

---

## Overview

This project presents an end-to-end artificial intelligence system for detecting rice leaf diseases using:

- Convolutional Neural Network (EfficientNetB0) for image classification  
- Natural Language Processing (NLP) module for human-readable explanations  
- Reinforcement Learning (RL) for threshold optimization  
- Grad-CAM for model interpretability  

The objective is to provide an accurate, explainable, and practical decision-support tool for agricultural use.

---

## Key Results

| Metric     | Value |
|------------|------|
| Accuracy   | 0.9935 |
| Macro-F1   | 0.9899 |

The model achieves high predictive performance while maintaining balanced classification across all disease categories.

---

## Ablation Study

| Experiment              | Accuracy | Macro-F1 |
|------------------------|---------:|---------:|
| Baseline (Full Model)  | 0.9935 | 0.9899 |
| No Augmentation        | 0.9928 | 0.9880 |
| No Class Weights       | 0.9915 | 0.9792 |

### Observations

- Data augmentation contributes to improved generalization and robustness  
- Class weighting has a stronger impact, particularly on macro-F1, indicating its importance in handling class imbalance  
- Removing class weights results in reduced performance for minority classes  

---

## Explainability (Grad-CAM)

Grad-CAM is used to visualize the regions of the input image that influence model predictions.

- Correct predictions show attention focused on disease-related regions  
- Misclassifications still highlight relevant leaf areas, indicating that errors arise from visual similarity rather than irrelevant features  

This supports the reliability and interpretability of the model.

---

## Features

- EfficientNetB0-based image classification with transfer learning  
- Class imbalance handling using class weights  
- Grad-CAM visualization for interpretability  
- NLP-based explanation module using TF-IDF  
- Reinforcement Learning for threshold optimization  
- Multi-metric evaluation including Accuracy and Macro-F1  

---

## Project Structure

```
Rice-Leaf-Disease-Detection/
│
├── data/
├── notebooks/
├── src/
│   ├── ablations/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── eval.py
│   └── rl_agent.py
│
├── experiments/
│   ├── results/
│   ├── logs/
│   └── configs/
│
├── docs/
│   ├── model_card.md
│   ├── ethics_statement.md
│   └── reports/
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-repo/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
```

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

This project uses the RiceLeafs dataset from Kaggle.

```bash
python data/get_data.py
```

---

## Training and Evaluation

Train main model:

```bash
python -m src.train
```

Evaluate main model:

```bash
python -m src.eval
```

---

## Ablation Experiments

```bash
python -m src.ablations.train_no_augmentation
python -m src.ablations.eval_no_augmentation

python -m src.ablations.train_no_class_weights
python -m src.ablations.eval_no_class_weights
```

---

## NLP Explanation Module

The NLP component uses TF-IDF vectorization and cosine similarity to generate short, human-readable explanations for each prediction.

---

## Reinforcement Learning Component

A Q-learning agent is used to optimize the classification confidence threshold. This improves decision-making without modifying the CNN model.

---

## Evaluation Metrics

- Accuracy  
- Macro-F1 Score  
- Confusion Matrix  

Macro-F1 is emphasized to ensure balanced performance across all classes.

---

## Inference (Using the Model)

After training the model, you can use it to predict a new image.

### Run the prediction script

```bash
python scripts/predict.py
```

### Input

You will be prompted to enter the full path of an image:

```
Enter image path: path/to/image.jpg
```

### Output

The system will display:

- Predicted class  
- Confidence score  
- Class probabilities  

Example:

```
Prediction Result
Predicted Class: LeafBlast
Confidence: 98.25%

Class Probabilities:
BrownSpot: 0.01%
Healthy: 0.50%
LeafBlast: 98.25%
Hispa: 1.24%
```

---

## Notes

- Images must be in JPG, JPEG, or PNG format  
- Recommended input size: clear leaf image  
- Results may vary depending on lighting and image quality  

---

## Ethical Considerations

This system is intended as a decision-support tool and should not replace professional agricultural diagnosis.

Limitations include:

- sensitivity to image quality and environmental conditions  
- potential dataset bias  
- limited number of disease classes  

Transparency is supported through Grad-CAM and explanation outputs.

See:

docs/ethics_statement.md

---

## Team Members

- Alejandro, Francine Angela G.  
- Decilio, Yohanna A.  
- Mendoza, Shane S.  
- Pascua, Maria Niña Grace L.  

---

## Summary

This project integrates multiple AI components:

- CNN for classification  
- NLP for explanation  
- RL for optimization  
- Grad-CAM for interpretability  

The result is a robust and interpretable system for rice leaf disease detection.