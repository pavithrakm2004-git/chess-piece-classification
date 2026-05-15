import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess Piece Classifier",
    page_icon="♟️",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

bundle        = load_model()
model         = bundle["model"]
scaler        = bundle["scaler"]
selector      = bundle["selector"]
labels_mapping = bundle["labels_mapping"]   # {0: 'b_bishop', ...}

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img_gray: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_gray, (64, 64))

    X_flat         = img_resized.flatten().astype(np.float32)
    mean_intensity = X_flat.mean()
    std_intensity  = X_flat.std()

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    gx = cv2.filter2D(img_resized.astype(np.float32), -1, sobel_x)
    gy = cv2.filter2D(img_resized.astype(np.float32), -1, sobel_y)
    edge_strength = np.sqrt(gx**2 + gy**2).mean()

    color_label = 1.0 if mean_intensity > 127 else 0.0
    color_black = 1.0 - color_label
    color_white = color_label

    engineered = np.array([mean_intensity, std_intensity, edge_strength,
                            color_label, color_black, color_white], dtype=np.float32)

    combined          = np.concatenate([X_flat, engineered]).reshape(1, -1)
    combined_scaled   = scaler.transform(combined)
    combined_selected = selector.transform(combined_scaled)
    return combined_selected


def predict(img_gray: np.ndarray):
    features  = extract_features(img_gray)
    pred_idx  = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]
    label     = labels_mapping[pred_idx]
    return label, proba, pred_idx


# ── Helpers ───────────────────────────────────────────────────────────────────
PIECE_EMOJI = {
    "bishop": "♗", "king": "♔", "knight": "♘",
    "pawn": "♙",  "queen": "♛", "rook": "♖",
}

def format_label(raw: str) -> str:
    color, piece = raw.split("_", 1)
    emoji       = PIECE_EMOJI.get(piece, "♟")
    color_name  = "White" if color == "w" else "Black"
    return f"{emoji} {color_name} {piece.capitalize()}"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("♟️ Chess Piece Classifier")
st.markdown("Upload an image of a chess piece to identify it instantly.")
st.divider()

uploaded = st.file_uploader(
    "Upload a chess piece image",
    type=["jpg", "jpeg", "png"],
    help="Colour or grayscale — the app converts automatically.",
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_np  = np.array(pil_img.convert("L"))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    label, proba, pred_idx = predict(img_np)
    color_code  = label.split("_")[0]
    confidence  = proba[pred_idx] * 100

    with col2:
        st.subheader("Result")
        st.markdown(f"## {format_label(label)}")
        st.metric("Confidence", f"{confidence:.1f}%")

        badge_bg   = "#f5f5f5" if color_code == "w" else "#1e1e1e"
        badge_text = "#000000" if color_code == "w" else "#ffffff"
        st.markdown(
            f'<div style="background:{badge_bg};color:{badge_text};'
            f'padding:10px;border-radius:8px;text-align:center;'
            f'font-size:1.1rem;font-weight:bold;margin-top:10px;">'
            f'{"⬜ White" if color_code=="w" else "⬛ Black"} Piece</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Top-3 Predictions")
    top3 = np.argsort(proba)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        lbl  = labels_mapping[idx]
        conf = proba[idx] * 100
        st.markdown(f"**{rank}. {format_label(lbl)}** — `{conf:.1f}%`")
        st.progress(conf / 100)

    with st.expander("All class probabilities"):
        rows = sorted(
            [(format_label(labels_mapping[i]), proba[i] * 100)
             for i in range(len(proba))],
            key=lambda x: -x[1],
        )
        for lbl_fmt, pct in rows:
            st.text(f"{lbl_fmt:<35} {pct:.2f}%")

else:
    st.info("👆 Upload a chess piece image to get started.")

st.divider()
st.caption("Model: Random Forest · 12 classes (6 pieces × Black / White)")
