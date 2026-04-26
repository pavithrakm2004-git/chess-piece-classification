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
