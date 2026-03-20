
# Rice Leaf Disease Detection using EfficientNet, NLP, and Reinforcement Learning

## Overview

This project develops an end-to-end AI system for detecting and classifying rice leaf diseases using deep learning, natural language processing, and reinforcement learning.

The system combines:

* a **Convolutional Neural Network (CNN)** for image classification
* a **Natural Language Processing (NLP)** module for explanation
* a **Reinforcement Learning (RL)** agent for decision optimization

The goal is to assist farmers and agricultural stakeholders in identifying rice leaf diseases quickly and accurately while also providing understandable explanations and improved decision thresholds.

---

# Features

* EfficientNetB0-based image classification (transfer learning)
* Class imbalance handling using class weights
* NLP-based explanation system using TF-IDF retrieval
* Reinforcement Learning (Q-learning) for threshold optimization
* Multi-metric evaluation (Accuracy, Macro-F1, Confusion Matrix)
* Modular and reproducible ML pipeline

---

# Project Structure

```
Rice-Leaf-Disease-Detection/
│
├── data/
│   ├── get_data.py
│   ├── raw/                  
│   └── processed/            
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_cnn.ipynb
│   ├── 04_nlp.ipynb
│   └── 05_rl.ipynb
│
├── src/
│   ├── data_pipeline.py
│   ├── data_pipeline_augmented.py
│   ├── models/
│   │   └── cnn_model.py
│   │   └── cnn_model_augmented.py
│   ├── train.py
│   ├── eval.py
│   ├── rl_agent.py
│   └── utils/
│       └── preprocess.py
│
├── experiments/
│   ├── results/
│   └── logs/
│   └── configs/
│
├── docs/
│   ├── proposal.pdf
│   ├── checkpoint.pdf
│   ├── final_report.pdf
│   ├── model_card.md
│   └── ethics_statement.md
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/your-repo/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
```

```bash
python -m venv venv
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

---

# Dataset

This project uses the **RiceLeafs dataset from Kaggle**.

## Setup Kaggle API

```bash
pip install kaggle
```

1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Account → API → Create New Token
3. Move `kaggle.json` to:

```
C:\Users\<username>\.kaggle\kaggle.json
```

## Download dataset

```bash
python data/get_data.py
```

---

# Data Preprocessing

```bash
python src/utils/preprocess.py
```

This will:

* clean corrupted images
* resize to 224×224
* split into train / validation
* generate CSV metadata

---

# Data Augmentation

```bash
python src/data_pipeline_augmentation.py
```

This will:

* spatial / geometric augmentation
* color / photometric augmentation
* fine-grained noise
* clipping

---


# Exploratory Data Analysis

Notebook:

```
notebooks/01_eda.ipynb
```

Includes:

* class distribution
* image visualization
* resolution analysis
* imbalance inspection

---

# Baseline Models

Notebook:

```
notebooks/02_baselines.ipynb
```

Models:

* Logistic Regression (with PCA)
* Simple CNN

Purpose:

* establish performance baseline

---

# CNN Model (Core Component)

Training:

```bash
python -m src.train
```

Evaluation:

```bash
python -m src.eval
```

Model:

* EfficientNetB0 (transfer learning)
* fine-tuning applied
* class weights for imbalance

### Example Performance

* Accuracy: ~0.40
* Macro-F1: ~0.35

Outputs:

```
experiments/results/
```

---

# NLP Explanation Module

Notebook:

```
notebooks/04_nlp.ipynb
```

Approach:

* TF-IDF vectorization
* cosine similarity retrieval
* disease knowledge base

### Example Output

```
Prediction: LeafBlast
Confidence: 0.78
Explanation: LeafBlast is the predicted class. Common symptoms include diamond-shaped lesions...
```

Purpose:

* improves interpretability
* provides user-friendly explanation

---

# Reinforcement Learning Component

Run:

```bash
python -m src.rl_agent
```

Notebook:

```
notebooks/05_rl.ipynb
```

Approach:

* Q-learning agent
* state = threshold
* actions = increase / decrease / maintain threshold
* reward = improvement in Macro-F1

### Results

```
CNN Base Macro-F1 : 0.2366
RL Best Macro-F1  : 0.2473
Best Threshold    : 0.40
```

RL successfully improves model performance by optimizing decision thresholds

---

# Evaluation Metrics

* Accuracy
* Macro-F1 Score
* Confusion Matrix

Macro-F1 is emphasized to ensure balanced performance across all disease classes.

---

# System Architecture

```
Image → CNN (EfficientNet)
          ↓
     Prediction + Confidence
          ↓
   RL Threshold Optimization
          ↓
 Final Decision → NLP Explanation
```

---

# Ethical Considerations

This system is intended as a **decision-support tool**, not a replacement for professional agricultural diagnosis.

Limitations:

* sensitive to image quality
* may not generalize to unseen disease variations
* environmental conditions may affect predictions

See:

```
docs/ethics_statement.md
```

---

# Team Members

* Alejandro, Francine Angela G.
* Decilio, Yohanna A.
* Mendoza, Shane S.
* Pascua, Maria Nina Grace L.

---

# Final Notes

This project demonstrates:

* deep learning for image classification
* NLP for explainability
* reinforcement learning for optimization

forming a complete, modular AI system.
