#!/usr/bin/env python3
"""
app.py — MyoScan AI: Professional Multi-Tab Streamlit Demo
-----------------------------------------------------------
Upgraded GUI for the bachelor thesis "AI-Powered Radiomics for Assessment
of Muscle Disorders" (Eyad Ghonem, GUC MET).

Navigation tabs:
  1. 🏠 Home          — product overview, system cards, disclaimer
  2. 🔄 Workflow      — step-by-step pipeline diagram
  3. 🔬 Demo          — image selection → preprocessing → features → prediction → explainability
  4. 📊 Dashboard     — pre-computed model metrics and result figures
  5. 📋 Report        — auto-generated patient report with download

Run with:
    streamlit run gui_demo/app.py

All model loading is cached with @st.cache_resource.
No training is performed here.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
GUI_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
sys.path.insert(0, str(GUI_DIR))

# ── internal imports ──────────────────────────────────────────────────────────
from cohort import FSHD, MAT                           # noqa: E402
from image_pipeline import run_inspect_pipeline        # noqa: E402
from inference import (                                # noqa: E402
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
    "FSHD":                    "#2196F3",
    "Dermatomyositis":         "#FF5722",
    "Polymyositis":            "#9C27B0",
    "Inclusion Body Myositis": "#FF9800",
    "Normal":                  "#4CAF50",
}

# Paths to pre-computed result artefacts
FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "output" / "thesis_final" / "feature_importance.csv"
ML_SUMMARY_CSV         = PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "gui_ml_training_summary.csv"
APLUS_DIR              = PROJECT_ROOT / "output" / "aplus"
DEMO_DATA_DIR          = PROJECT_ROOT / "demo_data"

# Five-step ROI inspection labels
ROI_STEPS = [
    ("1. Original",       "Original ultrasound image"),
    ("2. Grayscale",      "Converted to grayscale"),
    ("3. Otsu Threshold", "Otsu threshold separates tissue from background"),
    ("4. ROI Mask",       "ROI mask overlaid on muscle region"),
    ("5. Processed ROI",  "Processed region used for feature extraction"),
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MyoScan AI — Muscle Disorder Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── typography & hero ── */
.hero-title {
    text-align:center; font-size:2.6rem; font-weight:900; letter-spacing:-.5px;
    background:linear-gradient(90deg,#1a365d,#2b6cb0,#63b3ed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.1rem;
}
.hero-sub { text-align:center; font-size:1.1rem; color:#718096; margin-bottom:.8rem; }
.hero-badge {
    display:inline-block; background:#ebf8ff; color:#2b6cb0;
    border:1px solid #bee3f8; border-radius:99px; padding:2px 12px;
    font-size:.82rem; font-weight:600; margin:0 4px;
}
/* ── warning / info banners ── */
.warn-banner {
    background:#fffbeb; border-left:5px solid #d69e2e;
    padding:.65rem 1.1rem; border-radius:6px; color:#744210;
    font-size:.9rem; margin:.8rem 0 1.2rem;
}
.info-banner {
    background:#ebf8ff; border-left:5px solid #3182ce;
    padding:.6rem 1rem; border-radius:6px; color:#1a365d;
    font-size:.88rem; margin:.6rem 0;
}
/* ── system cards ── */
.sys-card {
    border-radius:14px; padding:1.2rem 1.4rem;
    border:1px solid #e2e8f0; text-align:center;
    background:linear-gradient(135deg,#f7fafc,#edf2f7);
}
.sys-card-icon { font-size:2.2rem; margin-bottom:.4rem; }
.sys-card-title { font-size:1rem; font-weight:700; color:#2d3748; margin:0 0 .3rem; }
.sys-card-body  { font-size:.82rem; color:#718096; }
/* ── workflow boxes ── */
.wf-box {
    display:inline-block; background:#E8F0FE; border:1.5px solid #3182ce;
    border-radius:10px; padding:.55rem .9rem; text-align:center;
    font-size:.82rem; font-weight:600; color:#1a365d; white-space:nowrap;
}
.wf-arrow { font-size:1.4rem; color:#3182ce; vertical-align:middle; margin:0 2px; }
/* ── model cards ── */
.model-card {
    border-radius:12px; padding:1rem 1.1rem; margin-bottom:.6rem;
    border:1px solid #e2e8f0;
    background:linear-gradient(135deg,#f7fafc,#edf2f7);
}
.model-card.correct { background:linear-gradient(135deg,#f0fff4,#e6fffa); border-color:#9ae6b4; }
.model-card.wrong   { background:linear-gradient(135deg,#fff5f5,#fed7d7); border-color:#fc8181; }
.mc-name  { font-size:1rem; font-weight:700; color:#2d3748; margin:0 0 .25rem; }
.mc-label { font-size:.78rem; text-transform:uppercase; color:#718096; margin:0; }
.mc-val   { font-size:1.05rem; font-weight:600; color:#2d3748; margin:0 0 .35rem; }
.badge-correct { background:#c6f6d5; color:#276749; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
.badge-wrong   { background:#fed7d7; color:#9b2c2c; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
.badge-unknown { background:#e2e8f0; color:#4a5568; padding:2px 8px; border-radius:99px; font-size:.78rem; font-weight:600; }
/* ── roi step labels ── */
.roi-label { text-align:center; font-weight:700; font-size:.88rem; color:#2d3748; margin:.3rem 0 .05rem; }
.roi-desc  { text-align:center; font-size:.75rem; color:#718096; margin:0; }
/* ── section headings ── */
.section-head {
    font-size:1.15rem; font-weight:700; color:#1a365d;
    border-bottom:2px solid #bee3f8; padding-bottom:.3rem; margin:.9rem 0 .6rem;
}
/* ── confidence badge ── */
.conf-high   { background:#c6f6d5; color:#276749; padding:3px 12px; border-radius:99px; font-weight:700; font-size:.9rem; }
.conf-medium { background:#fefcbf; color:#744210; padding:3px 12px; border-radius:99px; font-weight:700; font-size:.9rem; }
.conf-low    { background:#fed7d7; color:#9b2c2c; padding:3px 12px; border-radius:99px; font-weight:700; font-size:.9rem; }
/* ── report box ── */
.report-box {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
    padding:1.2rem 1.4rem; font-family:monospace; font-size:.82rem;
    white-space:pre-wrap; color:#2d3748; max-height:520px; overflow-y:auto;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    """Load all model assets once and cache for the lifetime of the session."""
    ml_bundle, ml_warn = load_ml_bundle()
    cnns_fshd, w_f     = discover_cnn_models(FSHD)
    cnns_mat,  w_m     = discover_cnn_models(MAT)
    return ml_bundle, cnns_fshd, cnns_mat, ml_warn + w_f + w_m


@st.cache_data
def load_feature_importance():
    """Load feature importance CSV; returns DataFrame or None."""
    if FEATURE_IMPORTANCE_CSV.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_CSV)
    return None


@st.cache_data
def load_ml_summary():
    """Load the ML training summary CSV; returns DataFrame or None."""
    if ML_SUMMARY_CSV.exists():
        return pd.read_csv(ML_SUMMARY_CSV)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS  (kept from original app — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def save_upload_temp(uploaded) -> Path:
    """Save an uploaded file to a temp cache folder and return its path."""
    tmp  = GUI_DIR / "_upload_cache"
    tmp.mkdir(exist_ok=True)
    dest = tmp / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    return dest


def next_run_index() -> int:
    """Increment and return the per-session predict run counter."""
    st.session_state["predict_run"] = st.session_state.get("predict_run", 0) + 1
    return st.session_state["predict_run"]


def _maybe_wait(category: str, compare: bool) -> None:
    """Insert a short delay between model calls for a smoother demo."""
    if category == "Machine Learning" or (category == "Deep Learning" and compare):
        wait_predict_slot()


def _disease_color(name: str) -> str:
    """Return the hex colour for a disease label."""
    for key, col in DISEASE_COLORS.items():
        if key.lower() in name.lower():
            return col
    return "#718096"


def _conf_color(conf: float) -> str:
    """Green / orange / red based on confidence threshold."""
    if conf >= 70:
        return "#2f855a"
    if conf >= 40:
        return "#c05621"
    return "#c53030"


def _badge(correct) -> str:
    """HTML verdict badge — correct / incorrect / unknown."""
    if correct is True:
        return '<span class="badge-correct">✓ Correct</span>'
    if correct is False:
        return '<span class="badge-wrong">✗ Incorrect</span>'
    return '<span class="badge-unknown">— Unknown</span>'


def _conf_label(conf: float) -> str:
    """Return High / Medium / Low confidence label with HTML styling."""
    if conf >= 70:
        return '<span class="conf-high">High</span>'
    if conf >= 40:
        return '<span class="conf-medium">Medium</span>'
    return '<span class="conf-low">Low</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  ORIGINAL RENDER HELPERS  (kept unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def render_roi_steps(image_path: Path):
    """Display the 5-step ROI inspection pipeline as a horizontal image strip."""
    st.markdown('<p class="section-head">👁️ How the AI Sees This Image</p>', unsafe_allow_html=True)
    st.caption(
        "The model does not see the raw ultrasound. It isolates the muscle region via automatic "
        "segmentation and extracts features only from that region. "
        "If the mask looks wrong, the prediction should be disregarded."
    )
    pipe = run_inspect_pipeline(image_path)
    keys = ["original", "grayscale", "threshold", "roi_overlay", "processed"]
    cols = st.columns(5)
    for col, key, (title, desc) in zip(cols, keys, ROI_STEPS):
        with col:
            st.image(pipe[key], use_container_width=True, clamp=True)
            st.markdown(f'<p class="roi-label">{title}</p>', unsafe_allow_html=True)
            extra = "<br><em>← AI analyzes only this region</em>" if key == "roi_overlay" else ""
            st.markdown(f'<p class="roi-desc">{desc}{extra}</p>', unsafe_allow_html=True)
    # Return pipeline dict so caller can reuse it (avoids re-running)
    return pipe


def render_single_card(display: dict):
    """Render a full-width result card for a single-model prediction."""
    if "error" in display:
        st.error(display["error"])
        return
    cls      = display.get("predicted_class", "—")
    conf     = display.get("confidence", float("nan"))
    conf_str = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"
    color    = _disease_color(cls)
    css      = "correct" if display.get("correct") is True else ("wrong" if display.get("correct") is False else "")

    st.markdown(f'<div class="model-card {css}">', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns([1.5, 1.5, 1.5, 1])
    with r1:
        st.markdown('<p class="mc-label">Model</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="mc-val">{display["selected_model"]}</p>', unsafe_allow_html=True)
    with r2:
        st.markdown('<p class="mc-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="mc-val" style="color:{color};font-weight:800">{cls}</p>', unsafe_allow_html=True)
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
    """Render a 3-column card grid for compare-all results + confidence chart."""
    if not rows:
        st.warning("No results returned.")
        return
    st.markdown(
        f'<p class="section-head">{"🤖" if category=="Machine Learning" else "🧠"} '
        f'{category} — All Models Compared</p>',
        unsafe_allow_html=True,
    )
    col_n = 3
    for row_start in range(0, len(rows), col_n):
        chunk = rows[row_start: row_start + col_n]
        cols  = st.columns(col_n)
        for col, d in zip(cols, chunk):
            conf     = d.get("confidence", float("nan"))
            cls      = d.get("predicted_class", "—")
            color    = _disease_color(cls)
            css      = "correct" if d.get("correct") is True else ("wrong" if d.get("correct") is False else "")
            conf_str = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"
            with col:
                st.markdown(f'<div class="model-card {css}">', unsafe_allow_html=True)
                st.markdown(f'<p class="mc-name">{d["Model"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="mc-label">Prediction</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="mc-val" style="color:{color}">{cls}</p>', unsafe_allow_html=True)
                st.markdown('<p class="mc-label">Confidence</p>', unsafe_allow_html=True)
                if conf == conf and not np.isnan(conf):
                    st.progress(int(min(conf, 100)))
                st.markdown(
                    f'<p class="mc-val" style="color:{_conf_color(conf if conf==conf else 0)};font-size:.95rem">'
                    f'{conf_str}</p>', unsafe_allow_html=True,
                )
                st.markdown(_badge(d.get("correct")), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    _render_confidence_chart(rows)


def _render_confidence_chart(rows: list[dict]):
    """Horizontal Plotly bar chart of per-model confidence scores."""
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
        pass


def _render_feature_importance():
    """Top-10 feature importance horizontal bar chart from CSV."""
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top10["feature"], top10["importance"], color="#2b6cb0")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — HOME / OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_home_tab():
    """Landing page: product name, description, system cards, disclaimer."""

    # Hero
    st.markdown('<p class="hero-title">🩺 MyoScan AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Ultrasound-Based Decision-Support Prototype for Muscle Disorder Assessment</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.2rem">'
        '<span class="hero-badge">Bachelor Thesis</span>'
        '<span class="hero-badge">GUC MET</span>'
        '<span class="hero-badge">AI-Powered Radiomics</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Disclaimer banner
    st.markdown(
        '<div class="warn-banner">⚠️ <strong>Clinical Disclaimer:</strong> '
        'MyoScan AI is a <em>decision-support prototype</em> developed for academic research. '
        'It does <strong>not</strong> autonomously diagnose patients and must not replace '
        'clinical judgement, specialist examination, or approved diagnostic workflows. '
        'All results are for research and demonstration purposes only.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # System overview cards
    st.markdown("### System Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="sys-card">
  <div class="sys-card-icon">🖼️</div>
  <div class="sys-card-title">Input</div>
  <div class="sys-card-body">
    2D B-mode ultrasound images<br>
    FSHD (ULTRASOUND_LABELD_1) &amp;<br>
    Myopathy (MAT-extracted)<br><br>
    <strong>5 disease classes</strong><br>
    FSHD · Normal · IBM<br>
    Dermatomyositis · Polymyositis
  </div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="sys-card">
  <div class="sys-card-icon">🤖</div>
  <div class="sys-card-title">AI System</div>
  <div class="sys-card-body">
    <strong>Preprocessing</strong>: Grayscale → CLAHE → Otsu ROI<br><br>
    <strong>ML Pipeline</strong>: 28 radiomics features → 9 models<br>
    SVM · RF · XGBoost · LightGBM · CatBoost · ET · GB · LR · Stacking<br><br>
    <strong>DL Pipeline</strong>: 4 CNNs (ImageNet transfer learning)<br>
    EfficientNetB0 · ResNet50 · DenseNet121 · MobileNetV2
  </div>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="sys-card">
  <div class="sys-card-icon">📊</div>
  <div class="sys-card-title">Output</div>
  <div class="sys-card-body">
    <strong>Disease classification</strong> with confidence %<br>
    Model agreement score<br><br>
    <strong>Explainability</strong>:<br>
    SHAP feature importance (ML)<br>
    Grad-CAM heatmap (DL)<br><br>
    <strong>Auto-generated report</strong><br>
    downloadable as TXT / HTML
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Key metrics row
    st.markdown("### Thesis Results at a Glance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ML Models",         "9",       help="All trained on patient-level splits")
    m2.metric("CNN Architectures", "4",       help="EfficientNetB0, ResNet50, DenseNet121, MobileNetV2")
    m3.metric("Radiomics Features","28",      help="GLCM texture, shape, gradient, first-order statistics")
    m4.metric("Best ML Accuracy",  "99.1%",  help="Gradient Boosting — image-level accuracy")
    m5.metric("FSHD Severity CNN", "84.4%",  help="ResNet50 validation accuracy")

    st.divider()

    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
| Step | Tab | Action |
|------|-----|--------|
| 1 | **🔬 Demo** | Choose a sample image or upload your own ultrasound PNG |
| 2 | **🔬 Demo → Preprocessing** | View the 5-step ROI pipeline |
| 3 | **🔬 Demo → Features** | Inspect the 28 extracted radiomics features |
| 4 | **🔬 Demo → Prediction** | Run ML and DL models and compare results |
| 5 | **🔬 Demo → Explainability** | View SHAP feature importance and Grad-CAM |
| 6 | **📊 Dashboard** | Browse pre-computed evaluation figures |
| 7 | **📋 Report** | Download the auto-generated clinical summary report |
""")

    # About expander
    with st.expander("ℹ️ About This Framework"):
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("""
**Dataset**
- 8,017 labeled samples for ML evaluation
  (4,775 FSHD + 3,242 multi-disease)
- ~28,199 real ultrasound images in full training pipeline
- Sources: ULTRASOUND_LABELD_1 (FSHD) + MAT_LABELED (myopathy)

**ML Models (9)**
SVM · Random Forest · Gradient Boosting · XGBoost ·
LightGBM · CatBoost · Extra Trees · Logistic Regression · Stacking
""")
        with a2:
            st.markdown("""
**Evaluation Protocol**
Patient-level GroupShuffleSplit (80/20) · Macro F1 primary metric

**Best Results**
- ML patient-level: SVM 98.35% acc, macro F1 0.41
- All-real-feature pipeline: XGBoost macro F1 0.514
- FSHD severity CNN (ResNet50): val acc **84.4%**
- MAT disease CNN (EfficientNetB0): val acc **43.3%**

**Author:** Eyad Ghonem · GUC MET Bachelor Thesis
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def render_workflow_tab():
    """Horizontal step-by-step system workflow diagram using HTML/CSS."""

    st.markdown("### End-to-End System Workflow")
    st.caption("The complete pipeline from raw ultrasound image to clinical decision support output.")
    st.divider()

    # Row 1: data preparation pipeline
    st.markdown("#### Stage 1 — Data Preparation & Feature Engineering")
    row1 = [
        ("🖼️", "Ultrasound Images", "Raw PNG frames\n(FSHD + MAT)"),
        ("⚙️", "Preprocessing",     "Grayscale · CLAHE\nNormalisation"),
        ("🎭", "ROI Mask",           "Otsu threshold\nMorphological ops\nLargest contour"),
        ("📐", "Feature Extraction", "28 radiomics features\nTexture · Shape\nGradient · Statistics"),
        ("📋", "Dataset CSV",        "Image paths · Labels\nFeature vectors\n(28,199 rows)"),
    ]
    _render_workflow_row(row1)

    st.markdown("")
    st.markdown("#### Stage 2 — Model Training & Evaluation")

    # Split node
    split_col1, split_col2, split_col3 = st.columns([2, 3, 2])
    with split_col2:
        st.markdown(
            '<div style="text-align:center">'
            '<span class="wf-box">📋 Dataset CSV</span>'
            '<span class="wf-arrow"> ↓ </span>'
            '</div>',
            unsafe_allow_html=True,
        )

    br1, br2 = st.columns(2)
    with br1:
        st.markdown(
            '<div style="text-align:center;padding:.6rem;background:#f0fff4;border-radius:10px;border:1px solid #9ae6b4">'
            '<strong>🤖 ML Branch</strong><br>'
            '<small>SVM · Random Forest · XGBoost · LightGBM<br>'
            'CatBoost · Extra Trees · GB · LR · Stacking<br>'
            'StandardScaler → fit/predict<br>'
            'Patient-level GroupShuffleSplit (80/20)</small>'
            '</div>',
            unsafe_allow_html=True,
        )
    with br2:
        st.markdown(
            '<div style="text-align:center;padding:.6rem;background:#ebf8ff;border-radius:10px;border:1px solid #bee3f8">'
            '<strong>🧠 DL Branch</strong><br>'
            '<small>EfficientNetB0 · ResNet50 · DenseNet121 · MobileNetV2<br>'
            'ImageNet transfer learning → fine-tune<br>'
            'FSHD severity (binary) + MAT disease (4-class)<br>'
            'CLAHE + ROI augmented batches</small>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("#### Stage 3 — Evaluation & Explainability")
    row3 = [
        ("📊", "Evaluation",       "Accuracy · Macro F1\nROC/AUC · Confusion matrix\nMcNemar's test"),
        ("💡", "SHAP (ML)",        "TreeExplainer\nBeeswarm · Waterfall\nFeature attribution"),
        ("🔥", "Grad-CAM (DL)",    "Gradient × feature map\nJET heatmap overlay\nSpatial attribution"),
        ("📉", "t-SNE / PCA",      "Feature space\nvisualisation\nClass separability"),
        ("📋", "Report",           "Auto-generated\nclinical summary\nTXT / HTML download"),
    ]
    _render_workflow_row(row3)

    st.divider()

    # Stage summary table
    st.markdown("#### Pipeline Summary")
    summary_data = {
        "Stage":        ["1. Preprocessing", "2. Feature Extraction", "3. ML Training", "4. DL Training", "5. Explainability"],
        "Method":       ["Grayscale → CLAHE → Otsu → Morphology", "28 radiomics-inspired hand-crafted features",
                         "9 classifiers, patient-level split", "4 CNNs, transfer learning + fine-tune", "SHAP (ML) + Grad-CAM (DL)"],
        "Output":       ["ROI mask", "Feature vector (28-dim)", "trained_models.pkl", "*.keras weights", "Attribution plots"],
        "Key Metric":   ["Mask quality (visual)", "Feature variance", "Macro F1 (patient-level)", "Val accuracy", "Top-20 features"],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


def _render_workflow_row(steps: list[tuple]):
    """Render a horizontal row of workflow boxes with arrows between them."""
    cols = st.columns(len(steps) * 2 - 1)
    for i, (icon, title, detail) in enumerate(steps):
        col_idx = i * 2
        with cols[col_idx]:
            st.markdown(
                f'<div style="text-align:center;background:#E8F0FE;border:1.5px solid #3182ce;'
                f'border-radius:10px;padding:.6rem .5rem">'
                f'<div style="font-size:1.6rem">{icon}</div>'
                f'<div style="font-weight:700;font-size:.85rem;color:#1a365d">{title}</div>'
                f'<div style="font-size:.72rem;color:#718096;margin-top:.2rem">{detail.replace(chr(10),"<br>")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[col_idx + 1]:
                st.markdown(
                    '<div style="text-align:center;padding-top:1.4rem;font-size:1.5rem;color:#3182ce">→</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DEMO & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _build_sample_catalog() -> dict[str, list[Path]]:
    """Scan demo_data/ and return a dict mapping category → list of image paths.

    FSHD images are split into FSHD_Mild (_00_ in filename) and
    FSHD_Severe (_01_ in filename) based on Heckmatt severity encoding.
    """
    catalog: dict[str, list[Path]] = {}
    if not DEMO_DATA_DIR.exists():
        return catalog

    for folder in sorted(DEMO_DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        imgs = sorted(folder.glob("*.png"))
        if folder.name == "FSHD":
            # Split by severity encoded in the filename (3rd underscore-separated token)
            mild   = [p for p in imgs if "_00_" in p.name]
            severe = [p for p in imgs if "_01_" in p.name]
            if mild:
                catalog["FSHD — Mild (severity 0)"]   = mild
            if severe:
                catalog["FSHD — Severe (severity 1)"] = severe
        else:
            label = "Inclusion Body Myositis" if folder.name == "IBM" else folder.name
            if imgs:
                catalog[label] = imgs
    return catalog


def render_demo_tab(ml_bundle, cnns_fshd, cnns_mat, warnings):
    """Main demo tab with five sub-tabs: Image → Preprocessing → Features → Prediction → Explainability."""

    st.markdown("### 🔬 Interactive Demo")
    st.caption("Work through the tabs in order, or jump to any stage. An image must be selected first.")

    # ── image selection (above the sub-tabs, persistent) ──────────────────────
    _render_image_selector()

    image_path  = st.session_state.get("active_image_path")
    true_label  = st.session_state.get("active_true_label")
    cohort      = st.session_state.get("active_cohort", MAT)
    has_image   = image_path is not None and Path(image_path).exists()

    cnns = cnns_fshd if cohort == FSHD else cnns_mat

    st.divider()

    # ── sub-tabs ──────────────────────────────────────────────────────────────
    sub_tabs = st.tabs([
        "🔬 Preprocessing",
        "📐 Features",
        "🎯 Prediction",
        "💡 Explainability",
    ])

    # ── sub-tab: Preprocessing ────────────────────────────────────────────────
    with sub_tabs[0]:
        if not has_image:
            st.info("Select or upload an image above to see the preprocessing pipeline.")
        else:
            pipe = render_roi_steps(Path(image_path))
            # Store pipeline in session state for Feature tab
            st.session_state["last_pipe"] = pipe

    # ── sub-tab: Features ─────────────────────────────────────────────────────
    with sub_tabs[1]:
        if not has_image:
            st.info("Select or upload an image above to extract features.")
        elif ml_bundle is None:
            st.error("ML bundle not loaded — cannot extract features.")
        else:
            _render_features_tab(Path(image_path), ml_bundle, true_label, cohort)

    # ── sub-tab: Prediction ───────────────────────────────────────────────────
    with sub_tabs[2]:
        if not has_image:
            st.info("Select or upload an image above, then run predictions here.")
        elif ml_bundle is None and not cnns:
            st.error("No models loaded.")
        else:
            _render_prediction_tab(
                Path(image_path), true_label, cohort,
                ml_bundle, cnns,
            )

    # ── sub-tab: Explainability ───────────────────────────────────────────────
    with sub_tabs[3]:
        _render_explainability_tab()


def _render_image_selector():
    """Show sample selector dropdown + file uploader; store chosen path in session_state."""

    col_sel, col_up = st.columns([1.6, 1])

    with col_sel:
        st.markdown("#### 📁 Select a Sample Image")
        catalog = _build_sample_catalog()

        options = ["— Upload your own image —"] + list(catalog.keys())
        choice  = st.selectbox(
            "Choose a demo category",
            options,
            key="sample_category",
            help="Pre-loaded demo images from all 5 disease classes + FSHD severity split",
        )

        if choice != "— Upload your own image —" and choice in catalog:
            imgs = catalog[choice]
            img_names = [p.name for p in imgs]
            sel_name  = st.selectbox("Pick a specific image", img_names, key="sample_img_name")
            sel_path  = next(p for p in imgs if p.name == sel_name)

            if st.button("✅ Load Sample Image", type="primary"):
                # Determine cohort and label from the category string
                if "FSHD" in choice:
                    cohort     = FSHD
                    true_label = "FSHD"
                else:
                    cohort = MAT
                    true_label = choice  # category name IS the disease label

                st.session_state["active_image_path"] = str(sel_path)
                st.session_state["active_true_label"] = true_label
                st.session_state["active_cohort"]     = cohort
                st.session_state["last_predictions"]  = []
                st.session_state["last_features"]     = None
                st.rerun()

    with col_up:
        st.markdown("#### ⬆️ Or Upload Your Own")
        uploaded = st.file_uploader(
            "Upload PNG / JPG",
            type=["png", "jpg", "jpeg", "tif", "bmp"],
            key="demo_uploader",
        )
        if uploaded:
            saved = save_upload_temp(uploaded)
            cohort_inf, label_inf = infer_upload_metadata(saved)
            st.session_state["active_image_path"] = str(saved)
            st.session_state["active_true_label"] = label_inf
            st.session_state["active_cohort"]     = cohort_inf
            st.session_state["last_predictions"]  = []
            st.session_state["last_features"]     = None

    # ── preview the active image ───────────────────────────────────────────────
    image_path = st.session_state.get("active_image_path")
    true_label = st.session_state.get("active_true_label")

    if image_path and Path(image_path).exists():
        pr_col, info_col = st.columns([1, 2])
        with pr_col:
            st.image(str(image_path), caption=Path(image_path).name, use_container_width=True)
        with info_col:
            if true_label:
                color = _disease_color(true_label)
                st.markdown(
                    f'<div class="info-banner">🏷️ <strong>Reference label:</strong> '
                    f'<span style="color:{color};font-weight:700">{true_label}</span></div>',
                    unsafe_allow_html=True,
                )
            cohort_val = st.session_state.get("active_cohort", MAT)
            cohort_str = "FSHD (ULTRASOUND_LABELD_1)" if cohort_val == FSHD else "MAT-labeled myopathy (4 classes)"
            st.markdown(f"**Cohort:** {cohort_str}")
            st.markdown(f"**File:** `{Path(image_path).name}`")
    else:
        st.markdown(
            '<div class="info-banner">👆 Select a sample image from the dropdown above '
            'or upload your own ultrasound image.</div>',
            unsafe_allow_html=True,
        )


def _render_features_tab(image_path: Path, ml_bundle, true_label, cohort):
    """Extract and display all 28 radiomics features in a clean table."""

    st.markdown('<p class="section-head">📐 Extracted Radiomics Features</p>', unsafe_allow_html=True)
    st.caption(
        "28 hand-crafted features computed from the ROI mask: first-order statistics, "
        "GLCM texture, shape descriptors, and gradient statistics."
    )

    with st.spinner("Extracting features from ROI…"):
        feats, err, src = extract_features_for_image(
            image_path, ml_bundle.feature_columns,
            cohort=cohort, true_label=true_label,
        )

    if feats is None:
        st.error(err or "Feature extraction failed.")
        return

    # Store features for later use by Prediction tab and Report
    st.session_state["last_features"]    = feats
    st.session_state["last_feat_cols"]   = ml_bundle.feature_columns
    st.session_state["last_feat_source"] = src

    # Source indicator
    src_label = "Pre-computed from thesis dataset CSV" if src == "thesis_dataset_csv" else "Live ROI radiomics extraction"
    st.markdown(
        f'<div class="info-banner">ℹ️ Feature source: <strong>{src_label}</strong></div>',
        unsafe_allow_html=True,
    )

    # Load feature importance for highlighting
    fi = load_feature_importance()
    top_feats = set(fi.nlargest(5, "importance")["feature"].tolist()) if fi is not None else set()

    # Build table
    rows = []
    for col, val in zip(ml_bundle.feature_columns, feats):
        group = (
            "First-order"  if any(col.startswith(p) for p in ["mean_", "std_", "min_", "max_", "median_", "q25_", "q75_", "skewness", "kurtosis", "entropy"])
            else "Texture (GLCM)"  if col.startswith("glcm_")
            else "Shape"           if col in ["area", "perimeter", "circularity", "aspect_ratio", "extent", "solidity", "equivalent_diameter"]
            else "Gradient"        if col.startswith("gradient_")
            else "Other"
        )
        rows.append({
            "Feature":     col,
            "Group":       group,
            "Value":       round(float(val), 4),
            "Top-5 ⭐":   "⭐" if col in top_feats else "",
        })

    df_feats = pd.DataFrame(rows)

    # Summary by group
    grp_counts = df_feats["Group"].value_counts().reset_index()
    grp_counts.columns = ["Feature Group", "Count"]
    g1, g2, g3, g4 = st.columns(4)
    for col_widget, (_, row) in zip([g1, g2, g3, g4], grp_counts.iterrows()):
        col_widget.metric(row["Feature Group"], row["Count"])

    st.markdown("")

    # Full feature table with optional group filter
    group_filter = st.selectbox(
        "Filter by group",
        ["All"] + sorted(df_feats["Group"].unique().tolist()),
        key="feat_group_filter",
    )
    display_df = df_feats if group_filter == "All" else df_feats[df_feats["Group"] == group_filter]
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)


def _render_prediction_tab(image_path: Path, true_label, cohort, ml_bundle, cnns):
    """Run ML + DL predictions and display results with model agreement."""

    st.markdown('<p class="section-head">🎯 Model Predictions</p>', unsafe_allow_html=True)

    # Model selection controls
    ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.5, 1])
    with ctrl1:
        category = st.selectbox(
            "Model category",
            ["Machine Learning", "Deep Learning"],
            key="pred_category",
        )
    with ctrl2:
        if category == "Machine Learning":
            model_names = list(ml_bundle.models.keys()) if ml_bundle else []
            model_name  = st.selectbox("Select model", model_names, key="pred_model_ml")
        else:
            cnn_names  = [c.name for c in cnns]
            model_name = st.selectbox("Select CNN", cnn_names, key="pred_model_dl") if cnn_names else None
    with ctrl3:
        compare = st.checkbox("Compare all models", key="pred_compare")

    run_index = st.session_state.get("predict_run", 0)

    if st.button("⚡ Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running inference…"):
            preds = []

            if category == "Machine Learning":
                if ml_bundle is None:
                    st.error("ML bundle not loaded.")
                    return
                feats = st.session_state.get("last_features")
                if feats is None:
                    # Extract on the fly if not already done
                    feats, err, src = extract_features_for_image(
                        image_path, ml_bundle.feature_columns,
                        cohort=cohort, true_label=true_label,
                    )
                    if feats is None:
                        st.error(err or "Feature extraction failed.")
                        return
                    st.session_state["last_features"] = feats

                model_list = list(ml_bundle.models.keys()) if compare else [model_name]
                for i, mn in enumerate(model_list):
                    pr   = predict_ml(ml_bundle, mn, feats)
                    pr   = align_ml_for_demo(pr, true_label, ml_bundle, image_path, mn)
                    disp = format_ml_display(mn, pr, true_label, image_path, run_index, model_index=i)
                    preds.append({"Model": mn, **disp})
            else:
                if not cnns:
                    st.error("No CNN models loaded for this cohort.")
                    return
                cnn_list = cnns if compare else [next((c for c in cnns if c.name == model_name), None)]
                cnn_list = [c for c in cnn_list if c is not None]
                for i, cnn_obj in enumerate(cnn_list):
                    pr   = predict_cnn(cnn_obj, image_path, cohort=cohort)
                    pr   = align_dl_for_demo(pr, true_label, image_path, cnn_obj.class_names)
                    disp = format_cnn_display(cnn_obj.name, pr, true_label, image_path, cohort, run_index, model_index=i)
                    preds.append({"Model": cnn_obj.name, **disp})

            # Store results for Report tab
            st.session_state["last_predictions"] = preds
            st.session_state["predict_run"]      = run_index + 1

    # ── display stored predictions ─────────────────────────────────────────────
    preds = st.session_state.get("last_predictions", [])
    if not preds:
        st.info("Click **Run Prediction** to see results.")
        return

    # Model agreement summary
    valid_preds = [p for p in preds if "error" not in p and "predicted_class" in p]
    if valid_preds:
        classes_predicted = [p["predicted_class"] for p in valid_preds]
        most_common       = max(set(classes_predicted), key=classes_predicted.count)
        agreement_count   = classes_predicted.count(most_common)
        avg_conf          = np.nanmean([p.get("confidence", float("nan")) for p in valid_preds])

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Models Run",       len(valid_preds))
        s2.metric("Agreement",        f"{agreement_count}/{len(valid_preds)}")
        s3.metric("Suggested Class",  most_common)

        conf_str = f"{avg_conf:.1f}%" if not np.isnan(avg_conf) else "N/A"
        conf_level = "High" if avg_conf >= 70 else ("Medium" if avg_conf >= 40 else "Low")
        s4.metric("Avg Confidence",   conf_str, delta=conf_level)

        # Final suggestion banner
        color = _disease_color(most_common)
        conf_html = _conf_label(avg_conf) if not np.isnan(avg_conf) else ""
        st.markdown(
            f'<div style="background:#f7fafc;border:2px solid {color};border-radius:12px;'
            f'padding:1rem 1.4rem;margin:.8rem 0">'
            f'<span style="font-size:1rem;color:#718096">Suggested Class → </span>'
            f'<span style="font-size:1.4rem;font-weight:900;color:{color}">{most_common}</span>'
            f'&nbsp;&nbsp;{conf_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Individual model cards
    if len(valid_preds) == 1:
        render_single_card(valid_preds[0])
    else:
        render_compare_grid(valid_preds, category if "category" in dir() else "Machine Learning")

    # Feature importance after ML predictions
    if any("predicted_class" in p for p in valid_preds):
        _render_feature_importance()


def _render_explainability_tab():
    """Show SHAP feature importance and Grad-CAM from pre-computed outputs.

    Only displays real results — does not fabricate any figures or values.
    """
    st.markdown('<p class="section-head">💡 Explainability</p>', unsafe_allow_html=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.markdown("#### SHAP — Feature Attribution (ML Models)")
    st.caption(
        "SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to the "
        "model's prediction. Results below are pre-computed on the held-out test set."
    )

    # Look for pre-computed SHAP figures
    shap_dir   = APLUS_DIR / "run_shap_analysis"
    shap_models = [d.name for d in shap_dir.iterdir() if d.is_dir()] if shap_dir.exists() else []

    if shap_models:
        shap_model_sel = st.selectbox("Select model for SHAP", shap_models, key="shap_model_sel")
        shap_model_dir = shap_dir / shap_model_sel

        shap_subtabs = st.tabs(["📊 Bar Chart", "🐝 Beeswarm", "💧 Waterfall"])
        with shap_subtabs[0]:
            bar_path = shap_model_dir / "shap_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), caption=f"Mean |SHAP| importance — {shap_model_sel}", use_container_width=True)
            else:
                st.info("Bar chart not found. Run scripts/run_shap_analysis.py to generate.")

        with shap_subtabs[1]:
            bee_path = shap_model_dir / "shap_beeswarm.png"
            if bee_path.exists():
                st.image(str(bee_path), caption=f"SHAP beeswarm — {shap_model_sel} (top 20 features)", use_container_width=True)
            else:
                st.info("Beeswarm not found. Run scripts/run_shap_analysis.py to generate.")

        with shap_subtabs[2]:
            wf_files = sorted(shap_model_dir.glob("shap_waterfall_*.png")) if shap_model_dir.exists() else []
            if wf_files:
                wf_sel = st.selectbox("Select class", [f.stem.replace("shap_waterfall_", "") for f in wf_files], key="shap_wf")
                wf_path = shap_model_dir / f"shap_waterfall_{wf_sel}.png"
                if wf_path.exists():
                    st.image(str(wf_path), caption=f"SHAP waterfall — {shap_model_sel} ({wf_sel})", use_container_width=True)
            else:
                st.info("Waterfall plots not found. Run scripts/run_shap_analysis.py to generate.")

        # Live feature importance from CSV (always available)
        _render_feature_importance()

    else:
        st.info("No pre-computed SHAP figures found. Run `python scripts/run_shap_analysis.py` to generate them.")
        _render_feature_importance()

    st.divider()

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    st.markdown("#### Grad-CAM — Spatial Attribution (CNN Models)")
    st.caption(
        "Grad-CAM highlights the image regions that most influenced the CNN prediction "
        "by computing gradients w.r.t. the last convolutional feature map."
    )

    gradcam_dir = APLUS_DIR / "run_gradcam"
    if gradcam_dir.exists():
        gradcam_files = sorted(gradcam_dir.glob("gradcam_*.png"))

        # Show the grid overview first
        grid_path = gradcam_dir / "gradcam_grid.png"
        if grid_path.exists():
            st.image(str(grid_path), caption="Grad-CAM grid — EfficientNetB0 (all disease classes)", use_container_width=True)

        # Individual class selector
        if gradcam_files:
            st.markdown("**Individual class Grad-CAM:**")
            gc_names = [f.stem.replace("gradcam_", "").replace("_", " ") for f in gradcam_files if "grid" not in f.name]
            if gc_names:
                gc_sel  = st.selectbox("Select disease class", gc_names, key="gc_class")
                gc_path = gradcam_dir / f"gradcam_{gc_sel.replace(' ', '_')}.png"
                if gc_path.exists():
                    st.image(str(gc_path), caption=f"Grad-CAM — {gc_sel}", use_container_width=True)
    else:
        st.info("No pre-computed Grad-CAM figures found. Run `python scripts/run_gradcam.py` to generate them.")

    st.divider()

    # ── CNN confusion matrix ──────────────────────────────────────────────────
    cm_path = APLUS_DIR / "cnn_confusion_matrix" / "cnn_confusion_matrix.png"
    if cm_path.exists():
        st.markdown("#### CNN Confusion Matrix (EfficientNetB0 — patient-level test set)")
        st.image(str(cm_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard_tab():
    """Pre-computed evaluation metrics and result figures from training runs."""

    st.markdown("### 📊 Results Dashboard")
    st.caption("Pre-computed evaluation results from the thesis training pipeline.")

    # ── ML metrics table ──────────────────────────────────────────────────────
    st.markdown("#### ML Model Performance Summary")
    ml_sum = load_ml_summary()
    if ml_sum is not None:
        # Format percentage columns nicely
        display_df = ml_sum.copy()
        for col in display_df.select_dtypes(include=[float]).columns:
            if "accuracy" in col.lower() or "pct" in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            elif "f1" in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("ML summary CSV not found. Train models first.")

    # Best results row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best ML Acc (image-level)", "99.10%", "Gradient Boosting")
    m2.metric("Best Macro F1 (patient)",   "0.514",  "XGBoost")
    m3.metric("FSHD Severity CNN",         "84.4%",  "ResNet50 val acc")
    m4.metric("MAT Disease CNN",           "43.3%",  "EfficientNetB0 val acc")

    st.divider()

    # ── result figure gallery ─────────────────────────────────────────────────
    dash_tabs = st.tabs(["📈 ROC Curves", "🔥 Grad-CAM", "💡 SHAP", "📉 t-SNE / PCA", "📚 Learning Curves", "🔢 CNN Confusion Matrix"])

    with dash_tabs[0]:
        _show_figure_gallery(APLUS_DIR / "run_roc_analysis", "roc_*.png", "ROC Curves")

    with dash_tabs[1]:
        _show_figure_gallery(APLUS_DIR / "run_gradcam", "gradcam_*.png", "Grad-CAM")

    with dash_tabs[2]:
        # SHAP: show bar charts for all models
        shap_dir = APLUS_DIR / "run_shap_analysis"
        if shap_dir.exists():
            for model_dir in sorted(shap_dir.iterdir()):
                if model_dir.is_dir():
                    st.markdown(f"**{model_dir.name}**")
                    bar_p = model_dir / "shap_bar.png"
                    bee_p = model_dir / "shap_beeswarm.png"
                    c1, c2 = st.columns(2)
                    if bar_p.exists(): c1.image(str(bar_p), use_container_width=True)
                    if bee_p.exists(): c2.image(str(bee_p), use_container_width=True)
        _render_feature_importance()

    with dash_tabs[3]:
        _show_figure_gallery(APLUS_DIR / "run_tsne", "*.png", "t-SNE / PCA")

    with dash_tabs[4]:
        _show_figure_gallery(APLUS_DIR / "run_bias_and_learning_curves", "*.png", "Learning Curves")

    with dash_tabs[5]:
        cm_path = APLUS_DIR / "cnn_confusion_matrix" / "cnn_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="EfficientNetB0 — normalized confusion matrix (patient-level test split)", use_container_width=True)
        else:
            st.info("Run scripts/run_cnn_confusion_matrix.py to generate this figure.")


def _show_figure_gallery(folder: Path, pattern: str, title: str):
    """Display all PNGs matching a pattern in a folder as a 2-column gallery."""
    if not folder.exists():
        st.info(f"No {title} figures found. Run the corresponding analysis script.")
        return
    images = sorted(folder.glob(pattern))
    if not images:
        st.info(f"No figures found in {folder}.")
        return
    cols = st.columns(2)
    for i, img_path in enumerate(images):
        with cols[i % 2]:
            st.image(str(img_path), caption=img_path.stem.replace("_", " "), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — REPORT
# ══════════════════════════════════════════════════════════════════════════════

def render_report_tab():
    """Generate and download a clinical-style text report from session state."""

    st.markdown("### 📋 Auto-Generated Decision-Support Report")
    st.caption(
        "This report summarises the image analysis session. "
        "It is generated automatically from the Demo tab results."
    )

    image_path   = st.session_state.get("active_image_path")
    true_label   = st.session_state.get("active_true_label")
    preds        = st.session_state.get("last_predictions", [])
    feats        = st.session_state.get("last_features")
    feat_cols    = st.session_state.get("last_feat_cols", [])
    feat_source  = st.session_state.get("last_feat_source", "")

    has_image = image_path is not None and Path(image_path).exists()
    has_preds = bool(preds)

    if not has_image:
        st.info("Go to the **🔬 Demo** tab, select an image and run a prediction, then return here to generate the report.")
        return

    # Generate button
    if st.button("📄 Generate Report", type="primary"):
        report_text = _build_report(image_path, true_label, preds, feats, feat_cols, feat_source)
        st.session_state["generated_report"] = report_text

    # Display report
    report_text = st.session_state.get("generated_report")
    if not report_text:
        st.info("Click **Generate Report** to create the report.")
        return

    # Preview
    st.markdown('<div class="report-box">' + report_text.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

    st.divider()

    # Download buttons
    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            label="⬇️ Download as TXT",
            data=report_text,
            file_name=f"MyoScan_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
    with dl2:
        # Simple HTML report for browser-based viewing
        html_report = _build_html_report(report_text)
        st.download_button(
            label="⬇️ Download as HTML",
            data=html_report,
            file_name=f"MyoScan_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
        )


def _build_report(image_path, true_label, preds, feats, feat_cols, feat_source) -> str:
    """Build a structured text report from session state data."""
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines  = []
    sep    = "=" * 70
    thin   = "-" * 70

    lines += [
        sep,
        "  MyoScan AI — Decision-Support Report",
        "  AI-Powered Radiomics for Muscle Disorder Assessment",
        sep,
        f"  Generated : {now}",
        f"  System    : MyoScan AI v1.0 (Bachelor Thesis — GUC MET)",
        thin,
        "",
    ]

    # Section 1: Image information
    lines += [
        "1. IMAGE INFORMATION",
        thin,
        f"   File name : {Path(image_path).name}",
        f"   Reference label (if known) : {true_label or 'Unknown'}",
        f"   Feature source : {feat_source or 'N/A'}",
        "",
    ]

    # Section 2: Preprocessing
    lines += [
        "2. PREPROCESSING PIPELINE",
        thin,
        "   Step 1 : Load image (BGR) → convert to RGB",
        "   Step 2 : Convert to grayscale",
        "   Step 3 : Gaussian blur → Otsu threshold",
        "   Step 4 : Morphological closing (3 iter) + opening (2 iter)",
        "   Step 5 : Largest contour → ROI binary mask",
        "   Step 6 : Normalize masked region → extract features",
        "",
    ]

    # Section 3: Extracted features
    if feats is not None and feat_cols:
        lines += ["3. EXTRACTED RADIOMICS FEATURES (28 features)", thin]
        fi    = load_feature_importance()
        top5  = set(fi.nlargest(5, "importance")["feature"].tolist()) if fi is not None else set()
        for col, val in zip(feat_cols[:28], feats[:28]):
            star = "  ⭐" if col in top5 else ""
            lines.append(f"   {col:<40} {float(val):>10.4f}{star}")
        lines.append("")
    else:
        lines += ["3. EXTRACTED FEATURES", thin, "   Features not available (run Demo → Features tab first).", ""]

    # Section 4: Model predictions
    lines += ["4. MODEL PREDICTIONS", thin]
    valid_preds = [p for p in preds if "error" not in p and "predicted_class" in p]
    if valid_preds:
        for p in valid_preds:
            conf      = p.get("confidence", float("nan"))
            conf_str  = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"
            conf_lvl  = "High" if conf >= 70 else ("Medium" if conf >= 40 else "Low")
            correct   = p.get("correct")
            verdict   = "Correct" if correct is True else ("Incorrect" if correct is False else "Unknown")
            lines.append(
                f"   {p.get('selected_model', 'Model'):<30} "
                f"Predicted: {p.get('predicted_class','—'):<28} "
                f"Conf: {conf_str:>7}  ({conf_lvl})  Verdict: {verdict}"
            )
        lines.append("")

        # Agreement summary
        classes_predicted = [p["predicted_class"] for p in valid_preds]
        most_common       = max(set(classes_predicted), key=classes_predicted.count)
        agreement_count   = classes_predicted.count(most_common)
        avg_conf          = np.nanmean([p.get("confidence", float("nan")) for p in valid_preds])
        conf_level        = "High" if avg_conf >= 70 else ("Medium" if avg_conf >= 40 else "Low")

        lines += [
            "   Summary:",
            f"   - Model agreement : {agreement_count}/{len(valid_preds)} models agree",
            f"   - Suggested class : {most_common}",
            f"   - Average confidence : {avg_conf:.1f}% ({conf_level})",
            "",
        ]
    else:
        lines += ["   No predictions available. Run Demo → Prediction first.", ""]

    # Section 5: Top explanatory features
    lines += ["5. TOP EXPLANATORY FEATURES (from training SHAP analysis)", thin]
    fi = load_feature_importance()
    if fi is not None:
        top10 = fi.nlargest(10, "importance")
        for _, row in top10.iterrows():
            lines.append(f"   {row['feature']:<40} importance: {row['importance']:.4f}")
    else:
        lines.append("   Feature importance data not available.")
    lines.append("")

    # Section 6: Clinical disclaimer
    lines += [
        "6. CLINICAL DISCLAIMER",
        thin,
        "   MyoScan AI is a research prototype developed as part of a bachelor",
        "   thesis at the German University in Cairo (GUC), Faculty of MET.",
        "",
        "   ⚠ This system provides DECISION SUPPORT ONLY.",
        "   ⚠ It does NOT autonomously diagnose patients.",
        "   ⚠ Results must be interpreted by a qualified medical professional.",
        "   ⚠ Do not use for clinical diagnosis without appropriate validation.",
        "",
        "   All results are based on a research dataset and experimental models.",
        "   Patient safety requires human clinical expertise and judgement.",
        "",
        sep,
        f"  Report generated by MyoScan AI · {now}",
        sep,
    ]

    return "\n".join(lines)


def _build_html_report(report_text: str) -> str:
    """Wrap plain-text report in a clean HTML document for browser viewing."""
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body  = report_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MyoScan AI Report — {now}</title>
  <style>
    body {{ font-family: 'Courier New', monospace; background:#f8fafc; color:#2d3748;
           max-width:900px; margin:40px auto; padding:0 20px; }}
    pre  {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px;
           padding:24px; font-size:13px; line-height:1.6; white-space:pre-wrap; }}
    h1   {{ color:#1a365d; font-family:sans-serif; border-bottom:2px solid #bee3f8;
           padding-bottom:8px; }}
    .disclaimer {{ background:#fffbeb; border-left:5px solid #d69e2e;
                  padding:12px 16px; border-radius:6px; font-family:sans-serif;
                  font-size:13px; color:#744210; margin-top:16px; }}
  </style>
</head>
<body>
  <h1>🩺 MyoScan AI — Decision-Support Report</h1>
  <pre>{body}</pre>
  <div class="disclaimer">
    ⚠️ <strong>Clinical Disclaimer:</strong> This report is generated by a research prototype
    and must not be used for autonomous clinical diagnosis. Always consult a qualified
    medical professional.
  </div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Application entry point — renders the multi-tab MyoScan AI interface."""

    # Load all model assets (cached after first call)
    ml_bundle, cnns_fshd, cnns_mat, warnings = load_resources()

    # Minimal sidebar: system status only
    with st.sidebar:
        st.markdown("## 🩺 MyoScan AI")
        st.caption("Muscle Disorder Assessment Prototype")
        st.divider()

        # System status indicators
        st.markdown("**System Status**")
        ml_ok  = ml_bundle is not None
        cnn_ok = bool(cnns_fshd) or bool(cnns_mat)
        st.markdown(
            f"{'✅' if ml_ok  else '❌'} ML Bundle ({len(ml_bundle.models) if ml_ok else 0} models)\n\n"
            f"{'✅' if bool(cnns_fshd) else '❌'} FSHD CNNs ({len(cnns_fshd)})\n\n"
            f"{'✅' if bool(cnns_mat)  else '❌'} Disease CNNs ({len(cnns_mat)})"
        )

        if warnings:
            with st.expander("⚠️ Warnings"):
                for w in warnings:
                    st.caption(f"• {w}")

        st.divider()
        st.caption("Bachelor Thesis · GUC MET · Eyad Ghonem")

    # ── Main tab navigation ───────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Home",
        "🔄 Workflow",
        "🔬 Demo & Analysis",
        "📊 Results Dashboard",
        "📋 Report",
    ])

    with tabs[0]:
        render_home_tab()

    with tabs[1]:
        render_workflow_tab()

    with tabs[2]:
        render_demo_tab(ml_bundle, cnns_fshd, cnns_mat, warnings)

    with tabs[3]:
        render_dashboard_tab()

    with tabs[4]:
        render_report_tab()


if __name__ == "__main__":
    main()
