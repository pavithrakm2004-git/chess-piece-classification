# ♟️ Chess Piece Classification using HOG + SVM

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![ML](https://img.shields.io/badge/ML-SVM%20%7C%20HOG-green)

## 📌 Problem Statement
Classify chess pieces (King, Queen, Rook, Bishop, Knight, Pawn — both white and black — 12 classes)
from segmented top-down chessboard square images. Preprocess by extracting individual squares and
applying illumination normalization. Extract HOG features for shape contours, aspect ratio and area
of the piece silhouette, and Hu moments for rotation-invariant shape description. Use PCA for
dimensionality reduction. Train multi-class SVM (one-vs-one), KNN, and Random Forest classifiers.
Evaluate per-piece classification accuracy and macro F1. Analyze how lighting variation and piece
design differences affect HOG feature stability.

## 📖 Project Overview
This project classifies chess pieces into **12 classes** (white/black × 6 piece types) from
top-down board images using **HOG features** and **SVM classifier**.

## 👥 Team
| Name | Role |
|------|------|
| **PAVITHRA K M** | Data Preparation & Preprocessing |
| **ANGEL GEORGE** | Feature Engineering & Dimensionality Reduction |
| **YADHU KRISHNAN V B** | Modeling, Evaluation & Analysis |

## 📂 Dataset
- **Source:** Kaggle (s4lman chess-pieces-dataset-85x85)
- **Classes:** 12
  - White: King, Queen, Rook, Bishop, Knight, Pawn
  - Black: King, Queen, Rook, Bishop, Knight, Pawn
- **Total images after augmentation:** 2400 (200 per class)

## ⚙️ Preprocessing Pipeline
1. Class audit and dataset selection
2. Color-based splitting: 6 original classes → 12 classes (white/black detection via brightness)
3. Grayscale conversion + resize to 64×64
4. CLAHE illumination normalization (clipLimit=2.0, tileGridSize=8×8)
5. Augmentation to balance classes (200 per class)

## 🧠 Feature Engineering
- **Statistical Features:** Mean intensity, standard deviation, edge strength (Sobel filter)
- **HOG Features:** Histogram of Oriented Gradients (winSize=64×64, blockSize=16×16, 9 bins)
- **Shape Features:** Contour area, aspect ratio from largest contour
- **Hu Moments:** 7 rotation-invariant shape descriptors
- **Color Feature:** Brightness-based white/black label encoding + one-hot encoding
- **Combined Feature Matrix:** HOG + statistical + shape features merged

## 📉 Dimensionality Reduction
- **StandardScaler** — feature normalization
- **PCA** — retaining 95% variance (`svd_solver='full'`)
- **SelectKBest** — top 500 features selected via ANOVA F-score

## 🤖 Models Trained
| Model | Description |
|-------|-------------|
| **SVM (RBF kernel)** | Multi-class, one-vs-one, with probability |
| **Random Forest** | 200 estimators |
| **Logistic Regression** | max_iter=1000 |

- Best model selected by **weighted F1-score**
- Overfitting/underfitting analyzed via train vs test accuracy gap

## 📊 Evaluation Metrics
- Per-class accuracy
- Weighted Precision, Recall, F1-score
- Confusion Matrix
- Train vs Test accuracy comparison

## Model Deployment
Model Deployed using Streamlit

App link: https://chess-classifier.streamlit.app

## Screenshots


<img width="1096" height="647" alt="Screenshot 2026-05-15 173201" src="https://github.com/user-attachments/assets/fc1cf151-09e7-4caa-9eb6-5359f2e02c54" />

<img width="1007" height="602" alt="Screenshot 2026-05-15 173629" src="https://github.com/user-attachments/assets/067b40a4-f2e9-489f-b3c4-2d750d843773" />

<img width="981" height="647" alt="Screenshot 2026-05-15 173133" src="https://github.com/user-attachments/assets/17578e8a-1d9d-48ff-8c55-710bf6ad0e84" />

<img width="1001" height="677" alt="Screenshot 2026-05-15 173809" src="https://github.com/user-attachments/assets/c2f83781-e29b-496a-802d-33c0f88fb970" />


## 📊 Results & Analysis
- Model analyzes how **lighting variation** and **piece design differences** affect HOG feature stability
- Top-3 predictions shown for each classification with confidence scores

## 🛠️ Tech Stack
- Python
- OpenCV
- Scikit-learn
- HOG (Histogram of Oriented Gradients)
- Streamlit



