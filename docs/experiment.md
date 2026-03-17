# Experiment Results of 02_baqseline.ipynb

## Dataset Overview
- Total training samples: 2684
- Total validation samples: 671
- Classes:
  - BrownSpot
  - Healthy
  - Hispa
  - LeafBlast

For baseline experiments, a subset was used:
- Training: 1000 samples
- Validation: 300 samples
- Image size: 224 × 224 × 3

---

## Baseline Models

### 1. Random Forest (Classical ML)
- Input: Flattened image pixels (150,528 features per image)
- Model: RandomForestClassifier (50 trees)

**Results:**
- Accuracy: **0.1567**
- Macro-F1: **0.1310**

---

### 2. Simple CNN (Deep Learning)
- Architecture:
  - 3 Convolutional layers (32, 64, 128 filters)
  - MaxPooling layers
  - Fully connected layer (128 units)
  - Dropout (0.3)
- Total parameters: ~11.17 million
- Training:
  - Epochs: 5
  - Batch size: 32
  - Optimizer: Adam

**Results:**
- Accuracy: **0.4100**
- Macro-F1: **0.2681**

---

## Comparison

| Model | Accuracy | Macro-F1 |
|------|------:|------:|
| Random Forest | 0.1567 | 0.1310 |
| Simple CNN | 0.4100 | 0.2681 |

---

## Observations

- The Simple CNN significantly outperformed the Random Forest model.
- Random Forest performed poorly due to loss of spatial information when flattening images.
- CNN performed better because it preserves spatial patterns and learns visual features directly.
- Despite improvement, CNN performance is still moderate, indicating:
  - complexity of the dataset
  - similarity between certain classes
- Confusion is likely between visually similar classes such as **Healthy and Hispa**, as observed during EDA.
- The relatively low Macro-F1 score suggests class imbalance is affecting performance.

---

## Key Insight

- Classical ML methods are not suitable for raw image classification tasks.
- CNN-based approaches are more effective but require stronger architectures.
- These baseline results justify the use of **EfficientNetB0** as the primary model in the next stage.

---

## Next Steps

- Implement EfficientNetB0 with transfer learning
- Improve performance using:
  - data augmentation
  - fine-tuning
- Evaluate using:
  - Accuracy
  - Macro-F1 Score
  - Confusion Matrix