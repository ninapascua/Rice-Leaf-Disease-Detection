# Ethics Statement  
Rice Leaf Disease Detection and Advisory System

---

## 1. Purpose

This system is designed to assist rice farmers and agricultural practitioners by detecting **four rice leaf conditions** (Brown Spot, Hispa, Leaf Blast, Healthy) using a convolutional neural network (EfficientNetB0).

The system also integrates:
- Natural Language Processing (NLP) for explanation generation  
- Grad-CAM for visual interpretability  
- Reinforcement Learning (RL) for decision threshold optimization  

The system is intended as a **decision-support tool** and **does not replace professional agricultural expertise**.

---

## 2. Ethical Risks

### 2.1 Misdiagnosis Risk

Incorrect predictions may lead to:
- inappropriate pesticide application  
- crop damage or yield loss  
- unnecessary financial costs  

Potential causes:
- variability in lighting conditions  
- partial or low-quality leaf images  
- visual similarity between diseases (e.g., Hispa vs other pest damage)  

#### Mitigation:

- display prediction confidence scores  
- provide Grad-CAM visual explanations  
- include NLP-based explanations for user understanding  
- clearly communicate system limitations  
- encourage consultation with agricultural experts  

---

### 2.2 Dataset Bias and Representativeness

The dataset used:
- is sourced from publicly available data (Kaggle RiceLeafs)  
- is not region-specific to Pampanga or the Philippines  
- contains relatively clean and controlled images  

This may result in:
- reduced performance in real-world farm conditions  
- bias toward image conditions present in the dataset  

#### Mitigation:

- use of **macro-F1 score** to evaluate performance across all classes  
- application of **class weighting** to reduce imbalance bias  
- validation through **ablation study**, confirming the importance of fairness mechanisms  
- planned expansion with locally collected field data  

---

### 2.3 Overreliance on AI Predictions

Users may treat system outputs as definitive decisions.

#### Mitigation:

The system includes a clear disclaimer:

> “This tool provides guidance only and should not replace professional agricultural diagnosis.”

Users are encouraged to:
- verify predictions with experts  
- consider environmental and contextual factors  
- use outputs as preliminary guidance only  

---

## 3. Transparency and Explainability

To promote trust and accountability, the system incorporates:

- **Grad-CAM visualization**, highlighting regions influencing predictions  
- **NLP explanations**, translating predictions into human-readable descriptions  
- full documentation of:
  - dataset sources  
  - preprocessing steps  
  - model architecture  
  - evaluation metrics  

### Observations from Grad-CAM:

- Correct predictions focus on disease-relevant leaf regions  
- Misclassifications still highlight meaningful areas, indicating that errors arise from visual similarity rather than arbitrary attention  

This supports the interpretability and reliability of the model.

---

## 4. Fairness Considerations

Class imbalance is addressed through:

- class weighting during training  
- evaluation using macro-F1 score  

### Ablation Findings:

- Removing class weights resulted in a significant drop in macro-F1  
- Minority classes (e.g., Healthy) showed reduced recall  

This demonstrates that fairness mechanisms are essential to ensure balanced performance across all classes.

---

## 5. Responsible Use Guidelines

Users should:

- treat predictions as advisory, not definitive  
- verify outputs with agricultural professionals  
- consider environmental and farm-specific conditions  
- avoid automated decision-making without human validation  

---

## 6. Limitations

- limited to four disease classes  
- dataset may not fully represent real-world conditions  
- performance may degrade under poor image quality  
- visually similar diseases may still be misclassified  

---

## 7. Future Ethical Improvements

- collect real-world field data from Pampanga and nearby regions  
- evaluate model performance under varying environmental conditions  
- incorporate human-in-the-loop validation  
- improve uncertainty estimation beyond softmax probabilities  
- extend fairness evaluation across additional datasets  

---

## 8. Conclusion

This system aims to balance **accuracy, fairness, and transparency** through the integration of CNN, NLP, RL, and explainability techniques.

While it demonstrates strong performance, it is designed to be used responsibly, with awareness of its limitations and appropriate human oversight.