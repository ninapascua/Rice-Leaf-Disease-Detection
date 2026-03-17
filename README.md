# Rice Leaf Disease Detection using EfficientNet

## Overview
This project develops a Convolutional Neural Network (CNN)-based system for detecting and classifying rice leaf diseases from images. The system uses EfficientNet for image classification and integrates explainability and decision-support components.

The goal of this project is to assist farmers and agricultural stakeholders in identifying rice leaf diseases quickly and accurately through automated image analysis.

## Features
- CNN-based image classification using EfficientNet
- Grad-CAM visualization for model explainability
- NLP-based module that provides disease descriptions
- Reinforcement Learning (RL) agent for threshold optimization
- Multi-metric performance evaluation (Accuracy, Macro-F1 Score, Confusion Matrix)

---

# Project Structure

```
Rice-Leaf-Disease-Detection/
│
├── data/
│   ├── get_data.py        # script to download dataset from Kaggle
│   ├── raw/               # raw dataset (ignored in GitHub)
│   └── processed/         # processed images
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_cnn.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── nlp_module.py
│   └── rl_agent.py
│
├── docs/
│   ├── checkpoint_report.md
│   ├── model_card.md
│   └── ethics_statement.md
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

This project uses the **RiceLeafs dataset from Kaggle**.

Before downloading the dataset, you must configure the Kaggle API.

---

## Step 1 — Install Kaggle API

```bash
pip install kaggle
```

---

## Step 2 — Get Kaggle API Key

1. Go to https://www.kaggle.com
2. Click your **profile icon**
3. Open **Account Settings**
4. Scroll to **API**
5. Click **Create New Token**

This downloads a file called:

```
kaggle.json
```

---

## Step 3 — Place kaggle.json in the correct folder

Move the downloaded file to:

```
C:\Users\<your-username>\.kaggle\kaggle.json
```

Example:

```
C:\Users\franc\.kaggle\kaggle.json
```

Create the `.kaggle` folder manually if it does not exist.

⚠️ **Important:**  
Never upload `kaggle.json` to GitHub because it contains your private API key.

---

## Step 4 — Download the dataset

Run:

```bash
python data/get_data.py
```

The dataset will be downloaded and extracted to:

```
data/raw/
```

---

# Data Preprocessing

To prepare the dataset, run:

```bash
python src/preprocess.py
```

This script will:

- remove corrupted images
- resize images to **224×224**
- create **train / validation / test splits**
- generate dataset CSV files

Generated files:

```
data/train.csv
data/val.csv
data/test.csv
```

---

# Exploratory Data Analysis

Open:

```
notebooks/01_eda.ipynb
```

This notebook analyzes:

- class distribution
- sample images
- image resolution distribution
- class imbalance

---

# Baseline Models

Run baseline experiments in:

```
notebooks/02_baselines.ipynb
```

Includes:

- Random Forest / SVM baseline
- simple CNN baseline

Evaluation metrics:

- Accuracy
- Macro-F1 Score
- Confusion Matrix

---

# CNN Model Training

EfficientNet experiments are implemented in:

```
notebooks/03_cnn.ipynb
```

Features include:

- EfficientNetB0 transfer learning
- data augmentation
- training curves
- Grad-CAM visualization

---

# NLP Explanation Module

The NLP module provides explanations for predicted diseases.

File:

```
src/nlp_module.py
```

Example output:

```
Prediction: Rice Blast
Explanation: Rice blast causes diamond-shaped lesions with gray centers and can reduce crop yield if untreated.
```

---

# Reinforcement Learning Agent

A Q-learning agent is implemented to optimize classification confidence thresholds based on macro-F1 score.

File:

```
src/rl_agent.py
```

---

# Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Macro-F1 Score
- Confusion Matrix

Macro-F1 ensures balanced evaluation across all classes.

---

# Ethical Considerations

This system is intended as a **decision-support tool** for farmers and agricultural stakeholders. It should not replace professional agricultural diagnosis.

Model predictions may be affected by image quality, lighting conditions, and unseen disease variations.

See:

```
docs/ethics_statement.md
```

Team Members:
Alejandro, Francine Angela G.
Decilio, Yohanna A.
Mendoza, Shane S.
Pascua, Maria Nina Grace L.