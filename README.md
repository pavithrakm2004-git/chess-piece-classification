# ♟️ Chess Piece Classification using HOG + SVM


## 📖 Project Overview
This project classifies chess pieces into **12 classes** (white/black × 6 piece types) from
top-down board images using **HOG features** and **SVM classifier**.

## 👥 Team

- **PAVITHRA K M**: Data Preparation & Preprocessing
- **ANGEL GEORGE**: Feature Engineering & Dimensionality Reduction  
- **YADHU KRISHNAN V B**: Modeling, Evaluation & Analysis

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

<img width="1040" height="548" alt="image" src="https://github.com/user-attachments/assets/a8a1fd14-9842-47ef-8a8d-54be098fee09" />

<img width="1007" height="602" alt="Screenshot 2026-05-15 173629" src="https://github.com/user-attachments/assets/e90bb543-24dc-45d9-9e13-8cf7e85a1457" />
<img width="1023" height="465" alt="Screenshot 2026-05-15 183215" src="https://github.com/user-attachments/assets/906b6554-70aa-4836-9d0c-2945ea730be4" />

<img width="928" height="577" alt="Screenshot 2026-05-15 172434" src="https://github.com/user-attachments/assets/f2b120c7-643e-4e36-9d49-442947e08c87" />
<img width="1037" height="452" alt="Screenshot 2026-05-15 182801" src="https://github.com/user-attachments/assets/5eaea2e7-981c-44df-a95a-99a7fbf51143" />

<img width="983" height="656" alt="Screenshot 2026-05-15 183746" src="https://github.com/user-attachments/assets/f1409b82-5d6c-44c3-93b7-768a5c2a3a38" />
<img width="986" height="448" alt="Screenshot 2026-05-15 183757" src="https://github.com/user-attachments/assets/1c287282-b116-4771-9dd5-5f8a6c036d66" />


## 📊 Results & Analysis
- Model analyzes how **lighting variation** and **piece design differences** affect HOG feature stability
- Top-3 predictions shown for each classification with confidence scores

## 🛠️ Tech Stack
- Python
- OpenCV (cv2)
- Scikit-learn (SVM, Random Forest, PCA, SelectKBest)
- HOG (Histogram of Oriented Gradients)
- Streamlit
- Matplotlib / Seaborn
- Pickle (model serialization) 



