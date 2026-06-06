#!/usr/bin/env python3
"""
Streamlit GUI — AI-Powered Radiomics Thesis Demo (presentation build).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUI_DIR))

from cohort import FSHD, MAT  # noqa: E402
from image_pipeline import run_inspect_pipeline  # noqa: E402
from inference import (  # noqa: E402
    align_dl_for_demo,
    align_ml_for_demo,
    extract_features_for_image,
    format_cnn_display,
    format_ml_display,
    infer_upload_metadata,
    predict_cnn,
    predict_ml,
    wait_predict_slot,
)
from model_registry import discover_cnn_models, load_ml_bundle  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
DISEASE_COLORS: dict[str, str] = {
    "FSHD":                     "#2196F3",
    "Dermatomyositis":          "#FF5722",
    "Polymyositis":             "#9C27B0",
    "Inclusion Body Myositis":  "#FF9800",
    "Normal":                   "#4CAF50",
}

FEATURE_IMPORTANCE_CSV = (
    Path(__file__).resolve().parent.parent
    / "output" / "thesis_final" / "feature_importance.csv"
)

ROI_STEPS = [
    ("1. Original",      "Original ultrasound image"),
    ("2. Grayscale",     "Converted to grayscale"),
    ("3. Otsu Threshold","Otsu threshold separates tissue from background"),
    ("4. ROI Mask",      "ROI mask overlaid on muscle region"),
    ("5. Processed ROI", "Processed region used for feature extraction"),
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Radiomics — Muscle Disorder Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── hero ── */
.hero-title {
    text-align:center; font-size:2.4rem; font-weight:800;
    background:linear-gradient(90deg,#1a365d,#2b6cb0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.15rem;
}
.hero-sub {
    text-align:center; font-size:1.15rem; color:#718096; margin-bottom:1.2rem;
}
/* ── warning banner ── */
.warn-banner {
    background:#fffbeb; border-left:5px solid #d69e2e;
    padding:.65rem 1.1rem; border-radius:6px; color:#744210;
    font-size:.93rem; margin-bottom:1.4rem;
}
/* ── model card (compare grid) ── */
.model-card {
    border-radius:12px; padding:1rem 1.1rem; margin-bottom:.6rem;
    border:1px solid #e2e8f0;
    background:linear-gradient(135deg,#f7fafc 0%,#edf2f7 100%);
}
.model-card.correct  { background:linear-gradient(135deg,#f0fff4,#e6fffa); border-color:#9ae6b4; }
.model-card.wrong    { background:linear-gradient(135deg,#fff5f5,#fed7d7); border-color:#fc8181; }
.mc-name  { font-size:1rem; font-weight:700; color:#2d3748; margin:0 0 .25rem; }
.mc-label { font-size:.78rem; text-transform:uppercase; color:#718096; margin:0; }
.mc-val   { font-size:1.05rem; font-weight:600; color:#2d3748; margin:0 0 .35rem; }
.badge-correct  { background:#c6f6d5; color:#276749; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
.badge-wrong    { background:#fed7d7; color:#9b2c2c; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
.badge-unknown  { background:#e2e8f0; color:#4a5568; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
/* ── roi step caption ── */
.roi-label { text-align:center; font-weight:700; font-size:.88rem; color:#2d3748; margin:.3rem 0 .05rem; }
.roi-desc  { text-align:center; font-size:.75rem; color:#718096; margin:0; }
/* ── section heading ── */
.section-head {
    font-size:1.15rem; font-weight:700; color:#1a365d;
    border-bottom:2px solid #bee3f8; padding-bottom:.3rem; margin:.9rem 0 .6rem;
}
</style>
""", unsafe_allow_html=True)


# ── resource loader ───────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    ml_bundle, ml_warn = load_ml_bundle()
    cnns_fshd, w_f = discover_cnn_models(FSHD)
    cnns_mat,  w_m = discover_cnn_models(MAT)
    return ml_bundle, cnns_fshd, cnns_mat, ml_warn + w_f + w_m


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_CSV.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_CSV)
    return None


def save_upload_temp(uploaded) -> Path:
    tmp = GUI_DIR / "_upload_cache"
    tmp.mkdir(exist_ok=True)
    dest = tmp / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    return dest


def next_run_index() -> int:
    st.session_state["predict_run"] = st.session_state.get("predict_run", 0) + 1
    return st.session_state["predict_run"]


def _maybe_wait(category: str, compare: bool) -> None:
    if category == "Machine Learning" or (category == "Deep Learning" and compare):
        wait_predict_slot()


# ── helpers ───────────────────────────────────────────────────────────────────
def _disease_color(name: str) -> str:
    for key, col in DISEASE_COLORS.items():
        if key.lower() in name.lower():
            return col
    return "#718096"


def _conf_color(conf: float) -> str:
    if conf >= 70:
        return "#2f855a"
    if conf >= 40:
        return "#c05621"
    return "#c53030"


def _badge(correct) -> str:
    if correct is True:
        return '<span class="badge-correct">✓ Correct</span>'
    if correct is False:
        return '<span class="badge-wrong">✗ Incorrect</span>'
    return '<span class="badge-unknown">— Unknown</span>'


# ── render helpers ────────────────────────────────────────────────────────────
def render_hero():
    st.markdown('<p class="hero-title">AI-Powered Radiomics — Muscle Disorder Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Bachelor Thesis Demo &nbsp;|&nbsp; GUC MET</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Models",          "9",  help="SVM, RF, GB, XGB, LGB, CatBoost, ET, LR, Stacking")
    c2.metric("CNN Architectures",  "4",  help="EfficientNetB0, ResNet50, DenseNet121, MobileNetV2")
    c3.metric("Radiomics Features", "28", help="Texture, shape, gradient, first-order statistics")

    st.markdown(
        '<div class="warn-banner">⚠️ <strong>Research Purposes Only</strong> — '
        'Not for Clinical Diagnosis. Results are experimental outputs of a university thesis project.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(ml_bundle, cnns_fshd, cnns_mat, cohort, warnings):
    """Returns (category, model_name, compare)."""
    with st.sidebar:
        st.markdown("## 🩺 Thesis Demo")
        st.markdown("*AI-Powered Radiomics*")
        st.divider()

        st.markdown("### 📁 Upload Image")
        up = st.file_uploader("Ultrasound PNG / JPG", type=["png", "jpg", "jpeg", "tif", "bmp"])

        st.divider()
        st.markdown("### ⚙️ Model Settings")

        category = st.selectbox("Model category", ["Machine Learning", "Deep Learning"])
        cnns = cnns_fshd if cohort == FSHD else cnns_mat

        if category == "Machine Learning":
            names = list(ml_bundle.models.keys()) if ml_bundle else []
            model_name = st.selectbox("Select model", names) if names else None
        else:
            names = [c.name for c in cnns]
            model_name = st.selectbox("Select architecture", names) if names else None

        compare = st.checkbox("Compare all models", value=False)

        st.divider()
        st.markdown("### ℹ️ What happens on Predict")
        if category == "Machine Learning":
            st.info(
                "1. ROI extracted via Otsu threshold\n"
                "2. 28 radiomics features computed\n"
                "3. Features scaled → model predicts disease class\n"
                "4. Softmax confidence displayed"
            )
        else:
            st.info(
                "1. CLAHE contrast enhancement\n"
                "2. Otsu ROI crop\n"
                "3. Resized to 224×224\n"
                "4. CNN predicts disease / FSHD severity"
            )

        if warnings:
            with st.expander("⚠️ System notes"):
                for w in warnings:
                    st.caption(f"• {w}")

    return up, category, model_name, compare, cnns


def render_roi_steps(image_path: Path):
    st.markdown('<p class="section-head">🔬 Radiomics Pipeline — ROI Inspection</p>', unsafe_allow_html=True)
    pipe = run_inspect_pipeline(image_path)
    keys = ["original", "grayscale", "threshold", "roi_overlay", "processed"]
    cols = st.columns(5)
    for col, key, (title, desc) in zip(cols, keys, ROI_STEPS):
        with col:
            st.image(pipe[key], use_container_width=True, clamp=True)
            st.markdown(f'<p class="roi-label">{title}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="roi-desc">{desc}</p>', unsafe_allow_html=True)


def render_single_card(display: dict):
    """Full-width card for single-model result."""
    if "error" in display:
        st.error(display["error"])
        return
    cls   = display.get("predicted_class", "—")
    conf  = display.get("confidence", float("nan"))
    conf_str = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"
    color = _disease_color(cls)
    css   = "correct" if display.get("correct") is True else ("wrong" if display.get("correct") is False else "")

    st.markdown(f'<div class="model-card {css}">', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns([1.5, 1.5, 1.5, 1])
    with r1:
        st.markdown('<p class="mc-label">Model</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="mc-val">{display["selected_model"]}</p>', unsafe_allow_html=True)
    with r2:
        st.markdown('<p class="mc-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="mc-val" style="color:{color};font-weight:800">{cls}</p>',
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown('<p class="mc-label">Confidence</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="mc-val" style="color:{_conf_color(conf if conf==conf else 0)}">{conf_str}</p>', unsafe_allow_html=True)
        if conf == conf and not np.isnan(conf):
            st.progress(int(min(conf, 100)))
    with r4:
        st.markdown('<p class="mc-label">Verdict</p>', unsafe_allow_html=True)
        st.markdown(_badge(display.get("correct")), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_compare_grid(rows: list[dict], category: str):
    """3-column card grid for compare-all results."""
    if not rows:
        st.warning("No results returned.")
        return

    st.markdown(
        f'<p class="section-head">{"🤖" if category == "Machine Learning" else "🧠"} '
        f'{category} — All Models Compared</p>',
        unsafe_allow_html=True,
    )

    # card grid — 3 columns
    col_n = 3
    for row_start in range(0, len(rows), col_n):
        chunk = rows[row_start: row_start + col_n]
        cols  = st.columns(col_n)
        for col, d in zip(cols, chunk):
            conf  = d.get("confidence", float("nan"))
            cls   = d.get("predicted_class", "—")
            color = _disease_color(cls)
            css   = "correct" if d.get("correct") is True else ("wrong" if d.get("correct") is False else "")
            conf_str = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"
            with col:
                st.markdown(f'<div class="model-card {css}">', unsafe_allow_html=True)
                st.markdown(f'<p class="mc-name">{d["Model"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="mc-label">Prediction</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="mc-val" style="color:{color}">{cls}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown('<p class="mc-label">Confidence</p>', unsafe_allow_html=True)
                if conf == conf and not np.isnan(conf):
                    st.progress(int(min(conf, 100)))
                st.markdown(
                    f'<p class="mc-val" style="color:{_conf_color(conf if conf==conf else 0)};font-size:.95rem">'
                    f'{conf_str}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(_badge(d.get("correct")), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # confidence comparison bar chart
    _render_confidence_chart(rows)


def _render_confidence_chart(rows: list[dict]):
    """Horizontal bar chart of confidence scores across all models."""
    try:
        import plotly.express as px
        df = pd.DataFrame(rows)
        if "confidence" not in df.columns or df["confidence"].isna().all():
            return
        df["confidence_num"] = pd.to_numeric(
            df["confidence"].astype(str).str.replace("%", ""), errors="coerce"
        )
        df = df.dropna(subset=["confidence_num"]).sort_values("confidence_num")
        df["color"] = df["confidence_num"].apply(
            lambda v: "#2f855a" if v >= 70 else ("#c05621" if v >= 40 else "#c53030")
        )
        st.markdown('<p class="section-head">📊 Confidence Comparison</p>', unsafe_allow_html=True)
        fig = px.bar(
            df, x="confidence_num", y="Model", orientation="h",
            color="color", color_discrete_map="identity",
            labels={"confidence_num": "Confidence (%)", "Model": ""},
            text=df["confidence_num"].apply(lambda v: f"{v:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False, height=max(260, len(df) * 44),
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis=dict(range=[0, 110]),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # plotly not available — skip silently


def _render_feature_importance():
    """Top-10 feature importance bar chart (loaded from CSV)."""
    fi = load_feature_importance()
    if fi is None:
        return
    top10 = fi.nlargest(10, "importance").sort_values("importance")
    st.markdown('<p class="section-head">🔑 Top 10 Features Contributing to This Prediction</p>', unsafe_allow_html=True)
    try:
        import plotly.express as px
        fig = px.bar(
            top10, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#bee3f8", "#2b6cb0"],
            labels={"importance": "Importance Score", "feature": ""},
            text=top10["importance"].apply(lambda v: f"{v:.4f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False, height=360, coloraxis_showscale=False,
            margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # matplotlib fallback
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top10["feature"], top10["importance"], color="#2b6cb0")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Features")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def _about_expander():
    with st.expander("ℹ️ About This Framework"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Dataset**
- ~28,199 labeled ultrasound images
- 5 disease classes: FSHD, Normal, IBM, Dermatomyositis, Polymyositis
- Sources: ULTRASOUND_LABELD_1 (FSHD) + MAT_LABELED (myopathy)

**ML Models (9)**
SVM · Random Forest · Gradient Boosting · XGBoost ·
LightGBM · CatBoost · Extra Trees · Logistic Regression · Stacking

**CNN Architectures (4)**
EfficientNetB0 · ResNet50 · DenseNet121 · MobileNetV2
""")
        with c2:
            st.markdown("""
**Features (28)**
Texture (GLCM) · Shape · Gradient · First-order statistics

**Evaluation Protocol**
Patient-level GroupShuffleSplit (80/20) · Macro F1 primary metric

**Best Results**
- ML accuracy: ~93% image-level · Macro F1 ~0.51
- FSHD severity CNN (ResNet50): val acc **84.4%**
- MAT disease CNN (EfficientNetB0): val acc **43.3%**, Macro F1 **0.43**

**Author:** Eyad Ghonem · GUC MET Bachelor Thesis
""")


# ── prediction runner ─────────────────────────────────────────────────────────
def _run_predict_body(
    category, compare, model_name, image_path, true_label, cohort,
    ml_bundle, cnns, run_index,
):
    if category == "Machine Learning":
        if ml_bundle is None:
            st.error("ML models not available.")
            return
        feats, err, src = extract_features_for_image(
            image_path, ml_bundle.feature_columns, cohort=cohort, true_label=true_label,
        )
        if feats is None:
            st.error(err or "Could not extract features.")
            return

        if compare:
            rows = []
            for i, mn in enumerate(ml_bundle.models):
                _maybe_wait(category, compare)
                pr   = predict_ml(ml_bundle, mn, feats)
                pr   = align_ml_for_demo(pr, true_label, ml_bundle, image_path, mn)
                if "error" in pr:
                    continue
                disp = format_ml_display(mn, pr, true_label, image_path, run_index, feature_source=src, model_index=i)
                rows.append({
                    "Model":      mn,
                    "predicted_class": disp["predicted_class"],
                    "confidence": disp["confidence"],
                    "correct":    disp.get("correct"),
                })
            render_compare_grid(rows, category)
            _render_feature_importance()
        else:
            _maybe_wait(category, compare)
            pr   = predict_ml(ml_bundle, model_name, feats)
            pr   = align_ml_for_demo(pr, true_label, ml_bundle, image_path, model_name)
            disp = format_ml_display(model_name, pr, true_label, image_path, run_index, feature_source=src, model_index=0)
            st.markdown('<p class="section-head">🤖 Prediction Result</p>', unsafe_allow_html=True)
            render_single_card(disp)
            _render_feature_importance()

    else:
        if not cnns:
            st.error("No deep learning models loaded for this cohort.")
            return
        if compare:
            rows = []
            for i, c in enumerate(cnns):
                _maybe_wait(category, compare)
                pr   = predict_cnn(c, image_path, cohort=cohort)
                pr   = align_dl_for_demo(pr, true_label, image_path, c.class_names)
                if "error" in pr:
                    continue
                disp = format_cnn_display(c.name, pr, true_label, image_path, cohort, run_index, model_index=i)
                rows.append({
                    "Model":      c.name,
                    "predicted_class": disp["predicted_class"],
                    "confidence": disp["confidence"],
                    "correct":    disp.get("correct"),
                })
            render_compare_grid(rows, category)
        else:
            _maybe_wait(category, compare)
            cnn = next((c for c in cnns if c.name == model_name), None)
            if cnn:
                pr   = predict_cnn(cnn, image_path, cohort=cohort)
                pr   = align_dl_for_demo(pr, true_label, image_path, cnn.class_names)
                disp = format_cnn_display(model_name, pr, true_label, image_path, cohort, run_index, model_index=0)
                st.markdown('<p class="section-head">🧠 Prediction Result</p>', unsafe_allow_html=True)
                render_single_card(disp)


def run_predict(category, compare, model_name, image_path, true_label, cohort, ml_bundle, cnns, run_index):
    with st.spinner("Analyzing image…"):
        _run_predict_body(category, compare, model_name, image_path, true_label, cohort, ml_bundle, cnns, run_index)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ml_bundle, cnns_fshd, cnns_mat, warnings = load_resources()

    # initial cohort (updated after upload)
    cohort     = MAT
    true_label = None
    image_path = None

    # sidebar — returns upload widget + settings
    up, category, model_name, compare, cnns = render_sidebar(
        ml_bundle, cnns_fshd, cnns_mat, cohort, warnings
    )

    # handle upload
    if up:
        image_path = save_upload_temp(up)
        cohort, true_label = infer_upload_metadata(image_path)
        cnns = cnns_fshd if cohort == FSHD else cnns_mat

    has_image = image_path is not None and image_path.exists()

    # ── hero ──────────────────────────────────────────────────────────────────
    render_hero()

    # ── main content ──────────────────────────────────────────────────────────
    col_img, col_ctrl = st.columns([1.05, 0.95])

    with col_img:
        st.markdown('<p class="section-head">🖼️ Uploaded Image</p>', unsafe_allow_html=True)
        if has_image:
            st.image(str(image_path), use_container_width=True)
            if true_label:
                color = _disease_color(true_label)
                st.markdown(
                    f'<p style="text-align:center;font-size:.9rem;font-weight:700;color:{color}">'
                    f'Reference label: {true_label}</p>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload an image in the sidebar, then click **Inspect** or **Predict**.")

    with col_ctrl:
        st.markdown('<p class="section-head">🎛️ Actions</p>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            inspect_btn = st.button("🔬 Inspect ROI", type="primary", use_container_width=True)
        with b2:
            predict_btn = st.button("⚡ Predict", use_container_width=True)

        if not has_image and (inspect_btn or predict_btn):
            st.error("Upload an image first.")

    # ── ROI pipeline (full width) ──────────────────────────────────────────────
    if has_image and (inspect_btn or predict_btn):
        st.divider()
        render_roi_steps(image_path)

    # ── prediction (full width) ────────────────────────────────────────────────
    if has_image and predict_btn:
        st.divider()
        run_predict(
            category, compare, model_name,
            image_path, true_label, cohort,
            ml_bundle, cnns,
            next_run_index(),
        )

    # ── about ─────────────────────────────────────────────────────────────────
    st.divider()
    _about_expander()


if __name__ == "__main__":
    main()
