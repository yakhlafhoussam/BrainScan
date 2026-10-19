"""
BrainScan AI — Streamlit Application

Upload a brain MRI image and receive a classification among four
categories: glioma, meningioma, no tumor, pituitary.

The model is a CNN trained on preprocessed MRI scans.
"""

import streamlit as st
import numpy as np
from PIL import Image

from predict import load_model, predict_image, CLASS_NAMES

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 880px; padding-top: 2.2rem; }

    .hero h1 { color: #17212b; font-size: 2.4rem; margin: 0 0 .25rem; }
    .hero p  { color: #61717f; font-size: 1.02rem; margin: 0; }

    .card {
        border: 1px solid #dbe4ea;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        background: #ffffff;
    }
    .card-label {
        color: #61717f; font-size: .72rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
    }
    .card-value {
        color: #17212b; font-size: 1.9rem; font-weight: 700;
        margin: .3rem 0 .1rem;
    }
    .card-sub { color: #087f8c; font-weight: 600; font-size: .95rem; }

    .disclaimer {
        color: #61717f; font-size: .78rem; margin-top: 2.5rem;
        border-top: 1px solid #dbe4ea; padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model Loading (cached once per process) ─────────────────────────
@st.cache_resource(show_spinner="Loading BrainScan model…")
def get_model():
    """Load and cache the trained Keras model."""
    return load_model()


try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(f"⚠️ Model loading failed:\n\n{exc}")
    st.stop()
except Exception as exc:
    st.error(f"⚠️ Unexpected error while loading the model:\n\n{exc}")
    st.stop()

# ── Header ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🧠 BrainScan AI</h1>
        <p>Upload a brain MRI image to see the model's prediction across four categories.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ── File Upload ─────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG, BMP",
)
st.caption("Categories: Glioma · Meningioma · No tumor · Pituitary tumor")

if uploaded_file is not None:
    # ── Read and validate the image ─────────────────────────────────
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
    except Exception:
        st.error(
            "❌ The uploaded file could not be read as an image. "
            "Please upload a valid JPG, JPEG, PNG, or BMP file."
        )
        st.stop()

    if image_rgb.size == 0 or image_rgb.ndim != 3:
        st.error(
            "❌ The uploaded image appears to be empty or has an unexpected format."
        )
        st.stop()

    # ── Run prediction ──────────────────────────────────────────────
    with st.spinner("Analyzing image…"):
        try:
            predicted_class, probabilities = predict_image(model, image_rgb)
        except Exception as exc:
            st.error(f"❌ Prediction failed:\n\n{exc}")
            st.stop()

    # ── Display results ─────────────────────────────────────────────
    confidence = probabilities[predicted_class]
    ranked_classes = sorted(probabilities, key=probabilities.get, reverse=True)
    runner_up = ranked_classes[1]
    display_class = DISPLAY_NAMES.get(predicted_class, predicted_class)

    st.divider()

    image_column, result_column = st.columns([1, 1], gap="large")
    with image_column:
        st.image(image_rgb, caption="Uploaded MRI image", use_container_width=True)

    with result_column:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Prediction</div>
                <div class="card-value">{display_class}</div>
                <div class="card-sub">{confidence:.1%} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if confidence < 0.70:
            st.warning(
                f"Low confidence — the next closest category is "
                f"{DISPLAY_NAMES[runner_up]} at {probabilities[runner_up]:.1%}."
            )
        else:
            st.info("This is a model estimate, not a diagnosis.")

    # ── Full distribution ───────────────────────────────────────────
    st.subheader("Class probabilities")

    table_rows = [
        {
            "Category": DISPLAY_NAMES[name],
            "Probability": probabilities[name] * 100,
        }
        for name in ranked_classes
    ]
    st.dataframe(
        table_rows,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", min_value=0, max_value=100, format="%.1f%%"
            )
        },
        hide_index=True,
        use_container_width=True,
    )

st.markdown(
    '<div class="disclaimer"><strong>Important:</strong> BrainScan AI is an experimental '
    "educational model and has not been clinically validated. It cannot replace assessment "
    "by a qualified medical professional.</div>",
    unsafe_allow_html=True,
)
