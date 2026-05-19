# ASL Vision

Real-time American Sign Language (ASL) character recognition system using deep learning, transfer learning, and computer vision.

---

# Project Goal

The goal of this project is to build a real-time ASL recognition system capable of recognizing:
- A–Z hand signs
- 0–9 digits

from webcam input using deep learning and computer vision techniques.

The system is designed as a practical accessibility-focused AI application combining:
- image classification
- transfer learning
- real-time inference
- webcam integration

---

# Planned Features

- ASL image classification
- Transfer learning with MobileNetV2
- Real-time webcam predictions
- OpenCV integration
- Prediction confidence scoring
- Confusion matrix evaluation
- Live demo interface

---

# Project Roadmap

## Phase 1 — Project Setup
- repository initialization
- environment setup
- folder structure
- dependency management

## Phase 2 — Dataset Exploration
- inspect class distributions
- visualize image samples
- analyze image dimensions
- preprocessing strategy

## Phase 3 — Data Preprocessing
- image resizing
- normalization
- train/validation split
- data augmentation

## Phase 4 — Model Training
- MobileNetV2 transfer learning
- baseline model training
- validation monitoring
- hyperparameter tuning

## Phase 5 — Model Evaluation
- accuracy/loss visualization
- confusion matrix
- misclassification analysis
- prediction inspection

## Phase 6 — Real-Time Inference
- webcam integration
- live frame preprocessing
- real-time predictions
- confidence overlays

## Phase 7 — Project Finalization
- README polishing
- architecture diagrams
- demo screenshots/GIFs
- GitHub optimization

---

# Planned Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Repository Structure

```text
asl-vision/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── webcam_app.py
│
├── models/
├── reports/
├── assets/
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```