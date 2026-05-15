import streamlit as st
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from PIL import Image
import io

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess Piece Classifier",
    page_icon="♟️",
    layout="wide",
)

# ─── Helpers ────────────────────────────────────────────────────────────────

def detect_color(img_gray):
    h, w = img_gray.shape
    center = img_gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return "w" if np.mean(center) > 127 else "b"


def preprocess_image(img_bgr):
    """Convert a BGR image to a processed 64×64 grayscale with CLAHE."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features_from_image(img):
    img_resized = cv2.resize(img, (64, 64))
    _, thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh[thresh > 0]) < 127:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = np.nan
    aspect_ratio = np.nan
    hu_moments = np.full(7, np.nan)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0:
            area = cv2.contourArea(largest)
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h != 0 else np.nan
            moments = cv2.moments(largest)
            hu_moments = cv2.HuMoments(moments).flatten()

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )
    hog_features = hog.compute(img_resized)
    hog_features = hog_features.flatten() if hog_features is not None else np.full(1764, np.nan)

    return np.concatenate([hog_features, [aspect_ratio], [area], hu_moments])


def load_images_from_folder(folder):
    X, y, labels_mapping = [], [], {}
    classes = sorted(os.listdir(folder))
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue
        labels_mapping[idx] = cls
        for f in os.listdir(cls_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(cls_path, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(img)
                    y.append(idx)
    return np.array(X), np.array(y), labels_mapping


def build_feature_matrix(X):
    X_flat = X.reshape(X.shape[0], -1)
    mean_intensity = X_flat.mean(axis=1)
    std_intensity  = X_flat.std(axis=1)

    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = sobel_x.T
    edge_strength = []
    for img in X:
        gx = cv2.filter2D(img.astype(np.float32), -1, sobel_x)
        gy = cv2.filter2D(img.astype(np.float32), -1, sobel_y)
        edge_strength.append(np.sqrt(gx**2 + gy**2).mean())
    edge_strength = np.array(edge_strength)

    color_feature = np.where(mean_intensity > 127, 'white', 'black')
    color_encoder = LabelEncoder()
    color_label   = color_encoder.fit_transform(color_feature)
    color_onehot  = pd.get_dummies(pd.Series(color_feature), prefix='color').to_numpy()

    engineered = np.column_stack([mean_intensity, std_intensity, edge_strength, color_label])
    X_combined = np.hstack([X_flat, engineered, color_onehot])
    return X_combined, color_encoder


def train_pipeline(data_dir, progress_cb=None):
    def step(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    step("Loading images…", 5)
    X, y, labels_mapping = load_images_from_folder(data_dir)

    step("Building feature matrix…", 20)
    X_combined, color_encoder = build_feature_matrix(X)

    step("Scaling features…", 35)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    step("Running PCA…", 45)
    pca   = PCA(n_components=0.95, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    step("Selecting best features…", 55)
    k = min(500, X_scaled.shape[1])
    selector   = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    step("Training models…", 65)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'SVC':                 SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model':                 model,
            'accuracy':              accuracy_score(y_test, y_pred),
            'precision':             precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall':                recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score':              f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_accuracy':        model.score(X_train, y_train),
            'test_accuracy':         accuracy_score(y_test, y_pred),
            'confusion_matrix':      confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'y_test':                y_test,
            'y_pred':                y_pred,
        }

    best_name  = max(results, key=lambda k: results[k]['f1_score'])
    best_model = results[best_name]['model']

    step("Saving model…", 90)
    pipeline = {
        'model_name':    best_name,
        'model':         best_model,
        'scaler':        scaler,
        'selector':      selector,
        'pca':           pca,
        'label_encoder': color_encoder,
        'labels_mapping': labels_mapping,
    }
    Path("output").mkdir(exist_ok=True)
    with open("output/best_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    step("Done!", 100)
    return results, best_name, labels_mapping, pca


def predict_single(img_bgr, pipeline):
    gray = preprocess_image(img_bgr)
    X    = np.array([gray])

    X_flat         = X.reshape(1, -1)
    mean_intensity = X_flat.mean(axis=1)
    std_intensity  = X_flat.std(axis=1)

    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = sobel_x.T
    gx = cv2.filter2D(gray.astype(np.float32), -1, sobel_x)
    gy = cv2.filter2D(gray.astype(np.float32), -1, sobel_y)
    edge_strength = np.sqrt(gx**2 + gy**2).mean()

    color_label  = pipeline['label_encoder'].transform(
        ['white' if mean_intensity[0] > 127 else 'black']
    )
    color_white  = np.array([[1 if mean_intensity[0] > 127 else 0]])
    color_black  = np.array([[0 if mean_intensity[0] > 127 else 1]])

    engineered   = np.column_stack([mean_intensity, std_intensity, [[edge_strength]], color_label])
    X_combined   = np.hstack([X_flat, engineered, color_white, color_black])

    X_scaled     = pipeline['scaler'].transform(X_combined)
    X_selected   = pipeline['selector'].transform(X_scaled)

    model        = pipeline['model']
    pred_idx     = model.predict(X_selected)[0]
    prob         = model.predict_proba(X_selected)[0] if hasattr(model, 'predict_proba') else None
    label        = pipeline['labels_mapping'][pred_idx]
    return label, prob, pred_idx


# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/ChessSet.jpg/320px-ChessSet.jpg",
                 use_column_width=True)
st.sidebar.title("♟️ Chess Piece Classifier")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🏋️ Train Model", "🔍 Predict"])

# ─── Home ───────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("♟️ Chess Piece Classifier")
    st.markdown("""
    Welcome! This app classifies chess pieces into **12 classes**:
    6 piece types × 2 colors (white / black).

    | Class | Meaning |
    |---|---|
    | `w_pawn` | White Pawn |
    | `b_pawn` | Black Pawn |
    | `w_rook` | White Rook |
    | `b_rook` | Black Rook |
    | … | … |

    ### How it works
    1. **Train** – point the app at your processed image folder and it will extract features, run PCA, select the best 500 features and train three classifiers (Logistic Regression, Random Forest, SVC).
    2. **Predict** – upload any chess piece image and get an instant prediction.

    ### Pipeline at a glance
    ```
    Raw image → Grayscale + CLAHE → Feature extraction
    (HOG · Sobel edges · intensity stats · shape moments)
    → StandardScaler → PCA (95 % variance) → SelectKBest (k=500)
    → Best model → Prediction
    ```
    """)

# ─── Train ──────────────────────────────────────────────────────────────────
elif page == "🏋️ Train Model":
    st.title("🏋️ Train the Classifier")

    data_dir = st.text_input(
        "Path to processed images folder",
        placeholder=r"C:\Users\you\chess-piece-classifier\data\processed",
        help="Folder containing sub-folders named like `w_pawn`, `b_rook`, …"
    )

    if st.button("🚀 Start Training", disabled=not data_dir):
        if not os.path.isdir(data_dir):
            st.error("Folder not found. Please check the path.")
        else:
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def progress_cb(msg, pct):
                progress_bar.progress(pct)
                status_text.info(f"**{msg}**")

            with st.spinner("Training in progress…"):
                try:
                    results, best_name, labels_mapping, pca = train_pipeline(data_dir, progress_cb)
                    st.success(f"✅ Training complete! Best model: **{best_name}**")

                    # ── Metrics table ──────────────────────────────────────
                    st.subheader("📊 Model Comparison")
                    summary = pd.DataFrame([
                        {
                            'Model':          name,
                            'Train Acc':      f"{s['train_accuracy']:.4f}",
                            'Test Acc':       f"{s['test_accuracy']:.4f}",
                            'Precision':      f"{s['precision']:.4f}",
                            'Recall':         f"{s['recall']:.4f}",
                            'F1-score':       f"{s['f1_score']:.4f}",
                        }
                        for name, s in results.items()
                    ])
                    st.dataframe(summary, use_container_width=True)

                    # ── Train vs Test bar chart ────────────────────────────
                    st.subheader("📈 Train vs Test Accuracy")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    model_names = list(results.keys())
                    train_accs  = [results[m]['train_accuracy'] for m in model_names]
                    test_accs   = [results[m]['test_accuracy']  for m in model_names]
                    x = np.arange(len(model_names))
                    ax.bar(x - 0.2, train_accs, 0.4, label='Train', color='steelblue')
                    ax.bar(x + 0.2, test_accs,  0.4, label='Test',  color='coral')
                    ax.set_xticks(x); ax.set_xticklabels(model_names)
                    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
                    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig)

                    # ── Overfitting analysis ──────────────────────────────
                    st.subheader("⚖️ Overfitting / Underfitting Analysis")
                    for name, s in results.items():
                        gap = s['train_accuracy'] - s['test_accuracy']
                        if gap > 0.05:
                            st.warning(f"**{name}**: gap={gap:.4f} → possible overfitting")
                        elif gap < -0.05:
                            st.warning(f"**{name}**: gap={gap:.4f} → possible underfitting")
                        else:
                            st.success(f"**{name}**: gap={gap:.4f} → balanced ✅")

                    # ── Confusion matrices ────────────────────────────────
                    st.subheader("🗂️ Confusion Matrices")
                    class_labels = [labels_mapping[i] for i in sorted(labels_mapping)]
                    for name, s in results.items():
                        st.markdown(f"**{name}**")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            s['confusion_matrix'], annot=True, fmt='d',
                            xticklabels=class_labels, yticklabels=class_labels,
                            cmap='Blues', ax=ax
                        )
                        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                        ax.set_title(f'Confusion Matrix — {name}')
                        plt.tight_layout()
                        st.pyplot(fig)

                    # ── PCA variance plot ─────────────────────────────────
                    st.subheader("📉 PCA Explained Variance")
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', markersize=3)
                    ax.set_xlabel('Components'); ax.set_ylabel('Cumulative variance')
                    ax.set_title('PCA — Explained Variance')
                    ax.grid(True)
                    st.pyplot(fig)

                    # ── Classification reports ────────────────────────────
                    st.subheader("📋 Classification Reports")
                    for name, s in results.items():
                        with st.expander(f"{name}"):
                            st.code(s['classification_report'])

                    st.info("💾 Model saved to `output/best_model.pkl`")

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.exception(e)

# ─── Predict ────────────────────────────────────────────────────────────────
elif page == "🔍 Predict":
    st.title("🔍 Predict a Chess Piece")

    model_path = st.text_input("Path to saved model (.pkl)", value="output/best_model.pkl")
    uploaded   = st.file_uploader("Upload a chess piece image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col1.image(pil_img, caption="Uploaded Image", use_column_width=True)

        if not os.path.exists(model_path):
            st.error("Model file not found. Please train first or check the path.")
        else:
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)

            # convert PIL → OpenCV BGR
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Classifying…"):
                label, prob, pred_idx = predict_single(img_bgr, pipeline)

            piece_color = "⬜ White" if label.startswith("w_") else "⬛ Black"
            piece_type  = label.split("_", 1)[1].capitalize()

            col2.markdown("### 🎯 Prediction")
            col2.markdown(f"**Class:** `{label}`")
            col2.markdown(f"**Color:** {piece_color}")
            col2.markdown(f"**Piece:** {piece_type}")

            if prob is not None:
                col2.markdown("### 📊 Class Probabilities")
                labels_mapping = pipeline['labels_mapping']
                prob_df = pd.DataFrame({
                    'Class':       [labels_mapping[i] for i in range(len(prob))],
                    'Probability': prob,
                }).sort_values('Probability', ascending=False).reset_index(drop=True)
                col2.dataframe(prob_df.head(6), use_container_width=True)

                # bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                top = prob_df.head(6)
                ax.barh(top['Class'][::-1], top['Probability'][::-1], color='steelblue')
                ax.set_xlabel('Probability')
                ax.set_title('Top-6 Class Probabilities')
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                plt.tight_layout()
                col2.pyplot(fig)

            # show preprocessed image
            gray_processed = preprocess_image(img_bgr)
            col1.image(gray_processed, caption="Preprocessed (64×64 CLAHE)", clamp=True)
