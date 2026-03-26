# Model Card — Rice Leaf Disease Detection System

---

## 1. Model Overview

This project presents a rice leaf disease detection system built using **EfficientNetB0** with transfer learning. The model classifies images of rice leaves into four categories and is supported by additional components that improve interpretability and decision-making.

The system integrates three AI components:

* **CNN (EfficientNetB0)** for image classification
* **NLP module** for human-readable explanations
* **Reinforcement Learning (RL)** for confidence threshold optimization

**Input:**
* RGB image resized to 224 × 224

**Output:**
* predicted class
* confidence score
* threshold-adjusted decision (via RL)
* explanation text

**Classes:**
* Brown Spot
* Hispa
* Leaf Blast
* Healthy

---

## 2. Intended Use

This system is designed as a **decision-support tool** for rice leaf disease identification.

It may be useful for:
* farmers and agricultural workers
* students and researchers
* agricultural monitoring applications

⚠️ The system is **not a replacement for expert diagnosis**. Predictions should be used as guidance only.

---

## 3. Training Data

The model was trained on a publicly available rice leaf disease dataset (Kaggle).

### Preprocessing:
* image resizing to 224 × 224
* normalization using EfficientNet preprocessing
* removal of corrupted images
* train-validation split (no data leakage)

### Class Distribution (Imbalanced):
* Leaf Blast — largest
* Healthy — smallest

To address imbalance:
* **class weights were applied**
* **macro-F1 was used as key metric**

---

## 4. Evaluation

### Metrics Used:
* Accuracy
* Macro-F1 Score
* Confusion Matrix

### Final Performance (Baseline Model)

| Metric     | Value   |
|------------|--------|
| Accuracy   | **0.9935** |
| Macro-F1   | **0.9899** |

The model achieves high performance while maintaining balanced classification across all classes.

---

## 5. Ablation Study

Two ablation experiments were conducted:

| Experiment              | Accuracy | Macro-F1 |
|------------------------|---------:|---------:|
| Baseline (Full Model)  | **0.9935** | **0.9899** |
| No Augmentation        | 0.9928 | 0.9880 |
| No Class Weights       | 0.9915 | 0.9792 |

### Findings:

* Removing augmentation slightly reduced performance → affects generalization
* Removing class weights caused a **larger drop in macro-F1**
  * especially reduced recall for minority classes (e.g., Healthy)

👉 This shows:
* augmentation improves robustness
* class weighting is critical for fairness

---

## 6. Reinforcement Learning Optimization

A **Q-learning agent** was used to optimize the classification confidence threshold.

### Results:

* Base Macro-F1: **0.9827**
* Best Threshold: **0.7**
* Improved Macro-F1: **0.9855**

This demonstrates that **post-processing decisions can improve model performance without retraining the CNN**.

---

## 7. Explainability (Grad-CAM)

To improve transparency, **Grad-CAM** was used to visualize model attention.

### Observations:

* Correct predictions → model focuses on lesion areas
* Misclassifications → still focuses on relevant disease regions

👉 This indicates:
* model learns meaningful features
* errors are due to **class similarity**, not random behavior

---

## 8. NLP Explanation Module

An NLP component generates short explanations for predictions using similarity-based retrieval.

### Example:

Prediction: Leaf Blast  
Explanation: Characterized by diamond-shaped lesions with gray centers caused by fungal infection.

This improves usability and interpretability for non-expert users.

---

## 9. Limitations

* limited to 4 disease classes
* dataset may not fully represent real-world conditions
* sensitive to lighting, blur, and angle variations
* visually similar diseases may cause misclassification

---

## 10. Ethical Considerations

### Risks:
* incorrect predictions may lead to wrong treatment
* over-reliance on AI outputs
* dataset bias affecting fairness

### Mitigation:
* use of macro-F1 to evaluate fairness
* class weighting to reduce bias
* Grad-CAM for transparency
* RL to refine decisions
* clear disclaimer for users

---

## 11. Future Work

* expand dataset (real farm data)
* support more disease types
* improve robustness
* deploy as mobile/web app
* integrate expert feedback

---

## 12. Summary

This system combines:

* CNN → classification
* NLP → explanation
* RL → optimization
* Grad-CAM → interpretability

Together, these components create a **robust, interpretable, and practical AI system** for rice leaf disease detection.