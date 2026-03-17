# Ethics Statement  
Rice Leaf Disease Detection and Advisory System

---

# 1. Purpose

This system supports rice farmers by detecting **four rice leaf conditions** (Brown Spot, Hispa, Leaf Blast, Healthy) using CNNs. It offers explanations through NLP and visual interpretability.

It is intended as a **support tool**, not a replacement for agricultural experts.

---

# 2. Ethical Risks

## 2.1 Misdiagnosis Risk
Wrong predictions may lead to:
- Incorrect pesticide use  
- Crop damage  
- Wasted resources  

Causes include:
- Lighting variability  
- Partial leaf images  
- Class ambiguity (e.g., Hispa vs other pest damage)

Mitigation:
- Show confidence score  
- Provide Grad-CAM explanations  
- Encourage expert consultation  

---

## 2.2 Dataset Bias
The RiceLeafs dataset:
- Is not region-specific  
- May not reflect Pampanga / PH agricultural conditions  
- Contains clean images without field noise  

Mitigation:
- Plan to expand dataset with local images  
- Incorporate field conditions in future versions  

---

## 2.3 Overreliance
Users may treat predictions as authoritative.

Mitigation:

> “This tool provides guidance only. Consult agricultural experts for final diagnosis.”

---

# 3. Transparency

Transparency tools included:
- Grad-CAM  
- TF-IDF explanation module  
- Public documentation of:
  - Dataset  
  - Preprocessing  
  - Splits  
  - Metrics  
  - Model architecture  

---

# 4. Responsible Use Guidelines

Users should:
- Verify disease with field experts  
- Use predictions as preliminary advice  
- Consider environmental or farm conditions  
- Not rely solely on automated output  

---

# 5. Future Ethical Work

- Gather field images from Pampanga / PH farms  
- Evaluate performance per class imbalance  
- Add human-in-the-loop verification  
- Introduce uncertainty estimates beyond softmax  

---