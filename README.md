# Chess Piece Classification using HOG + SVM

## Project Overview
Classifying chess pieces (12 classes: white/black × 6 piece types)
from top-down board images using HOG features and SVM classifier.

## Team
- **PAVITHRA K M**: Data Preparation & Preprocessing
- **ANGEL GEORGE**: Feature Engineering & Dimensionality Reduction  
- **YADHU KRISHNAN V B**: Modeling, Evaluation & Analysis

## Dataset
- Source: Kaggle (s4lman chess-pieces-dataset-85x85)
- Classes: 12 (w_king, w_queen, w_rook, w_bishop, w_knight, w_pawn,
            b_king, b_queen, b_rook, b_bishop, b_knight, b_pawn)
- Total images after augmentation: 2400 (200 per class)

## Preprocessing Pipeline (Member 1)
1. Class audit and dataset selection
2. Color-based splitting: 6 classes → 12 classes
3. Grayscale conversion + resize to 64×64
4. CLAHE illumination normalization
5. Augmentation to balance classes (200 per class)

## Project Structure
chess-piece-classifier/

├── data/

│   ├── raw/          (not tracked by git)

│   └── processed/    (not tracked by git)

├── notebooks/

│   └── 01_data_understanding.ipynb

├── src/

│   ├── preprocess.py

│   └── augment.py

├── requirements.txt

└── README.md


## Model Deployment
Model Deployed using Streamlit

App link: http://localhost:8501/

## Screenshots

<img width="1873" height="893" alt="Screenshot 2026-05-15 120259" src="https://github.com/user-attachments/assets/504007d4-5841-49cd-8c05-e1220ebb079d" />

<img width="1873" height="893" alt="Screenshot 2026-05-15 120259" src="https://github.com/user-attachments/assets/2ccf1403-0499-4978-987a-c5c9bcd3ad67" />

<img width="1898" height="879" alt="Screenshot 2026-05-15 120558" src="https://github.com/user-attachments/assets/ff25b93a-cd7b-458e-808e-31261884766f" />



