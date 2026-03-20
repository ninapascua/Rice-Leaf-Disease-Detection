# Model Card — Rice Leaf Disease Detection System

---

## 1. Model Overview

This project presents a rice leaf disease detection system built using **EfficientNetB0** with transfer learning. The model classifies images of rice leaves into four categories and is supported by additional components that improve interpretability and decision-making.

Alongside the CNN model, the system includes:

* a simple **NLP module** that provides short, human-readable explanations of predictions
* a **reinforcement learning (RL) component** that adjusts the confidence threshold to improve classification performance

The model is implemented using **TensorFlow/Keras**, with supporting tools from **Scikit-learn**.

**Input:**

* RGB image resized to 224 × 224

**Output:**

* predicted class
* confidence score
* threshold-adjusted decision (via RL)
* short textual explanation

**Classes:**

* Brown Spot
* Hispa
* Leaf Blast
* Healthy

---

## 2. Intended Use

The system is designed as a **decision-support tool** for identifying rice leaf diseases.

It may be useful for:

* farmers and agricultural workers
* students learning about plant diseases
* researchers exploring AI-based crop monitoring

The goal is to provide quick and accessible insights, especially in situations where expert consultation may not be immediately available.

However, the system is **not intended to replace professional diagnosis**. Its predictions should be used as guidance rather than final decisions.

---

## 3. Training Data

The model was trained using the **RiceLeafs dataset** from Kaggle.

The dataset contains images across four disease categories, with a few hundred samples per class. Most images were collected under relatively controlled conditions.

### Preprocessing steps included:

* removing corrupted or unreadable images
* resizing all images to 224 × 224
* applying EfficientNet-specific normalization
* splitting the data into training and validation sets

---

## 4. Evaluation

Model performance was evaluated using:

* **Accuracy**
* **Macro-F1 score**
* **Confusion matrix**

Macro-F1 was given particular importance since it accounts for class imbalance and evaluates performance across all categories more fairly.

### Baseline Comparison

| Model               | Accuracy   | Macro-F1   |
| ------------------- | ---------- | ---------- |
| Logistic Regression | ~0.20–0.25 | ~0.18–0.22 |
| Simple CNN          | ~0.30–0.35 | ~0.25–0.30 |
| EfficientNetB0      | ~0.40      | ~0.35      |

These results show that the EfficientNet-based model performs significantly better than simpler baselines.

---

## 5. Reinforcement Learning Optimization

A lightweight **Q-learning agent** was used to adjust the confidence threshold applied to predictions.

Instead of retraining the CNN, the RL agent learns which threshold leads to better classification performance based on validation results.

### Results

* CNN Base Macro-F1: **0.2366**
* RL Optimized Macro-F1: **0.2473**
* Best Threshold: **0.40**

Although the improvement is modest, it demonstrates that post-processing decisions can meaningfully enhance performance.

---

## 6. Explainability

To make the system easier to understand, an NLP-based explanation module is included.

Using TF-IDF and cosine similarity, the system retrieves relevant descriptions from a small knowledge base and generates a short explanation for each prediction.

### Example

Prediction: Leaf Blast
Explanation: Leaf Blast is characterized by diamond-shaped lesions with gray centers and is commonly associated with fungal infection.

This helps users better understand what the model is detecting, rather than relying solely on labels and scores.

---

## 7. Limitations

Several limitations should be considered:

* the model is trained on only four disease categories
* the dataset does not fully represent real-world farming conditions
* performance may drop on images with poor lighting, blur, or unusual angles
* class imbalance may still influence predictions

Because of these factors, results may vary when applied outside the dataset environment.

---

## 8. Ethical Considerations

### Potential Risks

* incorrect predictions could lead to improper treatment
* users may over-rely on automated results
* dataset bias may affect fairness and accuracy

### Mitigation

* confidence scores are provided with each prediction
* explanations are included to improve transparency
* RL is used to refine decision thresholds
* clear disclaimers emphasize that this is a support tool

Users are encouraged to verify results with agricultural experts whenever possible.

---

## 9. Future Work

Possible improvements include:

* collecting real-world data from local farms
* expanding the number of detectable diseases
* improving robustness to environmental variation
* deploying the system as a mobile or web application
* incorporating expert feedback into the system

---

## 10. Summary

This project demonstrates how multiple AI techniques can be combined into a single system:

* CNN for image classification
* NLP for interpretability
* RL for decision optimization

Together, these components create a more complete and practical solution for rice disease detection.

