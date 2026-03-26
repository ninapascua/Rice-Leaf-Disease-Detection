# Experiment Results

---

## 1. Dataset Overview

- Total training samples: 2684  
- Total validation samples: 671  

### Classes:
- BrownSpot  
- Healthy  
- Hispa  
- LeafBlast  

For baseline experiments, a subset was used:

- Training: 1000 samples  
- Validation: 300 samples  
- Image size: 224 × 224 × 3  

---

## 2. Baseline Models

### 2.1 Random Forest (Classical ML)

- Input: Flattened image pixels (150,528 features per image)  
- Model: RandomForestClassifier (50 trees)  

**Results:**

- Accuracy: 0.1567  
- Macro-F1: 0.1310  

---

### 2.2 Simple CNN (Deep Learning)

**Architecture:**
- 3 Convolutional layers (32, 64, 128 filters)  
- MaxPooling layers  
- Fully connected layer (128 units)  
- Dropout (0.3)  

- Total parameters: ~11.17 million  

**Training:**
- Epochs: 5  
- Batch size: 32  
- Optimizer: Adam  

**Results:**

- Accuracy: 0.4100  
- Macro-F1: 0.2681  

---

## 3. Baseline Comparison

| Model           | Accuracy | Macro-F1 |
|----------------|---------:|---------:|
| Random Forest  | 0.1567   | 0.1310   |
| Simple CNN     | 0.4100   | 0.2681   |

---

## 4. Baseline Observations

- The Simple CNN significantly outperformed the Random Forest model.  
- Random Forest performed poorly due to loss of spatial information when flattening images.  
- CNN performed better because it preserves spatial patterns and learns visual features directly.  
- Performance remained moderate, indicating dataset complexity and class similarity.  
- Confusion was observed between visually similar classes such as Healthy and Hispa.  
- Low Macro-F1 indicates class imbalance effects.  

---

## 5. Final Model: EfficientNetB0

To address baseline limitations, EfficientNetB0 with transfer learning was implemented.

### Key Improvements:

- Pretrained on ImageNet  
- Fine-tuning of top layers  
- Data augmentation (flip, rotation, contrast)  
- Class weighting to address imbalance  

---

## 6. Final Model Performance

| Metric   | Value |
|----------|------:|
| Accuracy | 0.9935 |
| Macro-F1 | 0.9899 |

### Interpretation:

- Significant improvement over baseline models  
- High accuracy with strong macro-F1 indicates balanced performance  
- Model successfully captures complex visual features  

---

## 7. Ablation Study

Two ablation experiments were conducted to evaluate the contribution of key components.

### Results:

| Experiment              | Accuracy | Macro-F1 |
|------------------------|---------:|---------:|
| Baseline (Full Model)  | 0.9935 | 0.9899 |
| No Augmentation        | 0.9928 | 0.9880 |
| No Class Weights       | 0.9915 | 0.9792 |

---

## 8. Ablation Analysis

### 8.1 Effect of Data Augmentation

- Slight decrease in performance without augmentation  
- Indicates augmentation improves generalization and robustness  

### 8.2 Effect of Class Weights

- Larger drop in Macro-F1 when removed  
- Reduced recall for minority classes (especially Healthy)  
- Demonstrates importance of handling class imbalance  

---

## 9. Key Insights

- Classical ML methods are ineffective for raw image classification  
- CNN-based models significantly improve performance  
- Transfer learning (EfficientNetB0) provides substantial gains  
- Data augmentation improves robustness  
- Class weighting is critical for fairness and balanced classification  

---

## 10. Conclusion

The experimental results demonstrate that EfficientNetB0, combined with augmentation and class weighting, achieves superior performance compared to baseline models.

The ablation study confirms that both augmentation and class weighting contribute to model performance, with class weighting having a greater impact on fairness and macro-F1.

This validates the design choices of the proposed system and supports its effectiveness for rice leaf disease classification.