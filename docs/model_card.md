# Model Card — Rice Leaf Disease Detection Model

---

# 1. Model Details

Model:
- EfficientNetB0 (transfer learning)

Framework:
- PyTorch / TensorFlow

Target Classes (4):
1. Brown Spot  
2. Hispa  
3. Leaf Blast  
4. Healthy  

Output:
- Predicted class  
- Confidence score  
- Evaluation metrics
- NLP-based explanation  
- Grad-CAM visualization  

---

# 2. Intended Use

This model assists with **rice leaf disease identification** for:

- Farmers  
- Students  
- Agricultural researchers  
- Extension workers  

Important:
**This tool provides advisory support and is not a replacement for expert diagnosis.**

---

# 3. Training Data

Dataset:
**RiceLeafs — Shayan Riyaz (Kaggle)**  
4 classes / several hundred images each.

Preprocessing:
- Corruption removal  
- Resizing (224×224)  
- Normalization  
- Train/val/test splits (70/15/15)

---

# 4. Evaluation Metrics

Primary metrics:
- Accuracy  
- Macro-F1  
- Confusion matrix  

Model results:

| Model         | Accuracy | Macro-F1 |
|---------------|----------|----------|
| SVM Baseline  | X.XX     | X.XX     |
| Simple CNN    | X.XX     | X.XX     |
| EfficientNetB0| X.XX     | X.XX     |

Grad-CAM is used to support interpretability.

---

# 5. Limitations

- Dataset limited to 4 diseases only  
- Possible lighting inconsistencies  
- Datasets under controlled environment 
- Not representative of all Philippine rice farm conditions  

Future improvements:
- Local PH datasets  
- Real field images  

---

# 6. Ethical Considerations

Risks:
- Misdiagnosis leading to wrong treatment  
- Overconfidence due to high accuracy  
- Dataset bias (non-PH images)

Mitigations:
- Display confidence score  
- Provide visual + text explanations  
- Include disclaimers  
- Encourage verification with experts  

---