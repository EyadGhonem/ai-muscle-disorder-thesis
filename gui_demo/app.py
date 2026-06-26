#!/usr/bin/env python3
"""
app.py  —  MyoScan AI  |  Premium Medical Decision-Support Demo
---------------------------------------------------------------
Multi-page Streamlit application for the bachelor thesis
"AI-Powered Radiomics for Assessment of Muscle Disorders"
(Eyad Ghonem, GUC MET).

Pages (controlled via st.session_state["page"]):
  splash      — full-screen hero intro
  welcome     — product overview
  workflow    — system pipeline diagram
  demo        — image selection, preprocessing, features, prediction, explainability
  dashboard   — thesis result metrics and figures
  comparison  — exploratory general-AI comparison
  report      — hospital-style A4 clinical report with HTML download

Run:
    streamlit run gui_demo/app.py
"""
from __future__ import annotations

import base64
import datetime
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── project paths ──────────────────────────────────────────────────────────
GUI_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
sys.path.insert(0, str(GUI_DIR))

from cohort import FSHD, MAT
from image_pipeline import run_inspect_pipeline
from inference import (
    align_dl_for_demo, align_ml_for_demo,
    extract_features_for_image, format_cnn_display, format_ml_display,
    infer_upload_metadata, predict_cnn, predict_ml, wait_predict_slot,
)
from model_registry import discover_cnn_models, load_ml_bundle

# ── constants ──────────────────────────────────────────────────────────────
DISEASE_COLORS = {
    "FSHD":                    "#8B1E3F",
    "Dermatomyositis":         "#B45309",
    "Polymyositis":            "#5B21B6",
    "Inclusion Body Myositis": "#0369A1",
    "Normal":                  "#065F46",
}
BRAND = {
    "burgundy":  "#8B1E3F",
    "navy":      "#1F2937",
    "grey_bg":   "#F6F7F9",
    "white":     "#FFFFFF",
    "light_red": "#FDECEF",
    "muted":     "#667085",
    "border":    "#E5E7EB",
    "green_bg":  "#D1FAE5", "green":  "#065F46",
    "amber_bg":  "#FEF3C7", "amber":  "#92400E",
    "red_bg":    "#FEE2E2", "red":    "#991B1B",
}
FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "output" / "thesis_final"   / "feature_importance.csv"
ML_SUMMARY_CSV         = PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "gui_ml_training_summary.csv"
APLUS_DIR              = PROJECT_ROOT / "output" / "aplus"
DEMO_DATA_DIR          = PROJECT_ROOT / "demo_data"
COMPARISON_CSV         = PROJECT_ROOT / "results" / "general_ai_comparison" / "test_cases.csv"
APLUS_FULL_DIR         = PROJECT_ROOT / "results" / "a_plus_full_improvements"

# Clinical interpretation text shown after prediction
CLINICAL_INTERPRETATION = {
    "Normal": (
        "The image features appear closer to the normal muscle ultrasound pattern "
        "in the training data. This does not exclude early or mild disease and should "
        "be interpreted with full clinical context."
    ),
    "Inclusion Body Myositis": (
        "The prediction suggests imaging patterns that may be consistent with inclusion "
        "body myositis. Key radiomics features may reflect muscle texture and structural "
        "changes associated with IBM. Specialist review, EMG, and biopsy remain essential."
    ),
    "IBM": (
        "The prediction suggests imaging patterns that may be consistent with inclusion "
        "body myositis. Specialist review, EMG, and biopsy remain essential."
    ),
    "Dermatomyositis": (
        "The prediction suggests patterns that may be consistent with dermatomyositis-"
        "related muscle involvement. Clinical symptoms, laboratory findings (CK, ANA), "
        "and specialist review are still required for diagnosis."
    ),
    "Polymyositis": (
        "The prediction suggests patterns that may be consistent with polymyositis-"
        "related muscle involvement. This output should support, not replace, clinical "
        "assessment including labs and muscle biopsy."
    ),
    "FSHD": (
        "The prediction suggests imaging patterns that may be consistent with FSHD-"
        "related muscle involvement. Severity assessment may support follow-up planning, "
        "but longitudinal clinical and genetic validation is required."
    ),
}

ROI_STEPS = [
    ("1. Original",       "Original ultrasound"),
    ("2. Grayscale",      "Grayscale conversion"),
    ("3. Otsu Threshold", "Otsu segmentation"),
    ("4. ROI Mask",       "Muscle region mask"),
    ("5. Processed ROI",  "Feature extraction region"),
]

PAGES = [
    ("welcome",    "1", "Home"),
    ("workflow",   "2", "Workflow"),
    ("demo",       "3", "Analysis"),
    ("dashboard",  "4", "Results"),
    ("comparison", "5", "AI Comparison"),
    ("report",     "6", "Report"),
    ("validation", "7", "Validation & Trust"),
]

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MyoScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── premium CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── global ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
/* Hide Streamlit top toolbar (Deploy button, hamburger menu) */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
.stDeployButton { display: none !important; }
.block-container { padding-top: 0.5rem !important; }

/* ── splash animation ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(28px); }
  to   { opacity: 1; transform: translateY(0);    }
}
.splash-anim { animation: fadeInUp .7s ease forwards; }

/* ── card ── */
.ms-card {
  background: #FFFFFF; border: 1px solid #E5E7EB;
  border-radius: 14px; padding: 20px 24px;
  box-shadow: 0 1px 6px rgba(0,0,0,.06);
  margin-bottom: 14px;
}
.ms-card-red {
  background: #FDECEF; border: 1.5px solid #8B1E3F;
  border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
}

/* ── section heading ── */
.ms-section {
  font-size: 1.05rem; font-weight: 700; color: #1F2937;
  border-left: 4px solid #8B1E3F; padding-left: 10px;
  margin: 20px 0 10px;
}
.ms-page-title {
  font-size: 1.7rem; font-weight: 800; color: #1F2937;
  letter-spacing: -.4px; margin-bottom: .2rem;
}
.ms-page-sub {
  font-size: .93rem; color: #667085; margin-bottom: 1rem;
}

/* ── navigation sidebar ── */
section[data-testid="stSidebar"] {
  background: #FFFFFF !important;
  min-width: 240px !important;
  width: 240px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  min-width: 240px !important;
  width: 240px !important;
}
section[data-testid="stSidebar"] .stButton > button {
  text-align: left !important;
  background: transparent !important;
  border: none !important;
  border-left: 3px solid transparent !important;
  border-radius: 0 8px 8px 0 !important;
  color: #374151 !important;
  font-weight: 500 !important;
  padding: 9px 14px 9px 16px !important;
  width: 100% !important;
  font-size: .875rem !important;
  box-shadow: none !important;
  line-height: 1.5 !important;
  white-space: nowrap !important;
  overflow: visible !important;
  min-height: unset !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: #FDECEF !important;
  color: #8B1E3F !important;
  border-left-color: #8B1E3F !important;
}
/* Active nav item via sibling marker */
section[data-testid="stSidebar"] .nav-active + div .stButton > button {
  background: #FDECEF !important;
  color: #8B1E3F !important;
  border-left: 3px solid #8B1E3F !important;
  font-weight: 700 !important;
}

/* ── Streamlit metric ── */
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E7EB;
  border-radius: 12px; padding: 14px 18px; }

/* ── primary button override ── */
.stButton > button[kind="primary"] {
  background: #8B1E3F !important; border-color: #8B1E3F !important;
  color: white !important; font-weight: 600 !important;
  border-radius: 8px !important; padding: 8px 22px !important;
}
.stButton > button[kind="primary"]:hover {
  background: #6D1730 !important; border-color: #6D1730 !important;
}

/* ── model result card ── */
.mc { border-radius:12px; padding:.9rem 1rem; margin-bottom:.5rem;
      border:1px solid #E5E7EB; background:#FFFFFF; }
.mc.ok  { background:#F0FDF4; border-color:#86EFAC; }
.mc.bad { background:#FEF2F2; border-color:#FCA5A5; }
.mc-label { font-size:.72rem; text-transform:uppercase; color:#667085; margin:0; }
.mc-val   { font-size:.98rem; font-weight:600; color:#1F2937; margin:0 0 .3rem; }
.badge-ok  { background:#DCFCE7;color:#166534;padding:2px 9px;border-radius:99px;font-size:.76rem;font-weight:600; }
.badge-bad { background:#FEE2E2;color:#991B1B;padding:2px 9px;border-radius:99px;font-size:.76rem;font-weight:600; }
.badge-unk { background:#F3F4F6;color:#374151;padding:2px 9px;border-radius:99px;font-size:.76rem;font-weight:600; }

/* ── ROI captions ── */
.roi-lbl { text-align:center;font-weight:700;font-size:.82rem;color:#1F2937;margin:.3rem 0 .05rem; }
.roi-dsc { text-align:center;font-size:.72rem;color:#667085;margin:0; }

/* ── disclaimer ── */
.disclaimer {
  background:#FDECEF; border:1.5px solid #8B1E3F; border-radius:10px;
  padding:14px 18px; font-size:.84rem; color:#1F2937; margin:.8rem 0;
}

/* ── A4 report wrapper ── */
.a4-wrap { background:#EBEBEB; padding:32px 20px; border-radius:12px; }
.a4-page {
  background:#FFFFFF; max-width:820px; margin:0 auto;
  border-radius:8px; box-shadow:0 4px 28px rgba(0,0,0,.14);
  overflow:hidden; font-family:'Inter',sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    ml_bundle, ml_warn = load_ml_bundle()
    cnns_fshd, w_f     = discover_cnn_models(FSHD)
    cnns_mat,  w_m     = discover_cnn_models(MAT)
    return ml_bundle, cnns_fshd, cnns_mat, ml_warn + w_f + w_m

@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_CSV.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_CSV)
    return None

@st.cache_data
def load_ml_summary():
    if ML_SUMMARY_CSV.exists():
        return pd.read_csv(ML_SUMMARY_CSV)
    return None

# ── check reportlab availability once ─────────────────────────────────────
try:
    import reportlab  # noqa: F401
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ══════════════════════════════════════════════════════════════════════════
#  NAVIGATION
# ══════════════════════════════════════════════════════════════════════════

def _go(page: str):
    """Navigate to a page, scroll to top, and rerun."""
    if st.session_state.get("page") != page:
        st.session_state["_scroll_top"] = True
    st.session_state["page"] = page
    st.rerun()


def _inject_scroll_behavior(scroll_now: bool = False):
    """Scroll main view to top on page change; also on Streamlit sub-tab clicks."""
    flag = "true" if scroll_now else "false"
    components.html(
        f"""
<script>
(function () {{
  function scrollTop() {{
    try {{
      const w = window.parent;
      w.scrollTo(0, 0);
      const doc = w.document;
      for (const sel of [
        'section.main',
        '[data-testid="stMain"]',
        '[data-testid="stAppViewContainer"]',
        '.main'
      ]) {{
        const el = doc.querySelector(sel);
        if (el) el.scrollTo(0, 0);
      }}
    }} catch (e) {{}}
  }}

  if ({flag}) {{
    scrollTop();
    setTimeout(scrollTop, 120);
  }}

  const w = window.parent;
  if (!w._myoscanScrollInstalled) {{
    w._myoscanScrollInstalled = true;
    w.document.addEventListener('click', function (e) {{
      const tab = e.target.closest('[data-baseweb="tab"]');
      if (tab) setTimeout(scrollTop, 80);
    }}, true);
  }}
}})();
</script>
        """,
        height=0,
        width=0,
    )


def render_sidebar_stepper():
    """Render the branded sidebar stepper navigation."""
    current = st.session_state.get("page", "welcome")
    with st.sidebar:
        # Logo
        st.markdown(f"""
<div style="padding:18px 16px 10px">
  <div style="display:flex;align-items:center;gap:10px">
    <svg width="32" height="32" viewBox="0 0 52 52" fill="none">
      <circle cx="26" cy="26" r="26" fill="#8B1E3F"/>
      <path d="M8 26 Q13 16 18 26 Q23 36 28 26 Q33 16 38 26 Q41 20 44 26"
            stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/>
      <circle cx="26" cy="26" r="3.5" fill="white" opacity=".9"/>
    </svg>
    <div>
      <div style="font-size:.95rem;font-weight:800;color:#1F2937;letter-spacing:-.3px">MyoScan AI</div>
      <div style="font-size:.7rem;color:#667085">GUC MET Thesis</div>
    </div>
  </div>
</div>
<div style="height:1px;background:#E5E7EB;margin:0 16px 12px"></div>
<div style="padding:0 8px 4px;font-size:.7rem;text-transform:uppercase;letter-spacing:.8px;color:#667085;font-weight:600">
  Navigation
</div>
""", unsafe_allow_html=True)

        for page_id, step_num, label in PAGES:
            is_active = page_id == current
            if is_active:
                st.markdown('<div class="nav-active"></div>', unsafe_allow_html=True)
            if st.button(f"  {step_num}   {label}", key=f"nav_{page_id}"):
                _go(page_id)

        st.markdown("""
<div style="height:1px;background:#E5E7EB;margin:12px 16px 12px"></div>
""", unsafe_allow_html=True)

        # System Status — check key paths without crashing
        with st.expander("System Status", expanded=False):
            checks = [
                ("ML bundle (.pkl)",
                 PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "trained_models.pkl"),
                ("ML summary CSV",      ML_SUMMARY_CSV),
                ("Feature importance",  FEATURE_IMPORTANCE_CSV),
                ("Demo data folder",    DEMO_DATA_DIR),
                ("SHAP analysis",       APLUS_DIR / "run_shap_analysis"),
                ("Grad-CAM output",     APLUS_DIR / "run_gradcam"),
                ("A+ results folder",   APLUS_FULL_DIR),
                ("CNN models folder",   GUI_DIR / "models"),
                ("Comparison CSV",      COMPARISON_CSV),
            ]
            for label, path in checks:
                found = Path(path).exists()
                icon = "🟢" if found else "🔴"
                st.caption(f"{icon} {label}")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: SPLASH
# ══════════════════════════════════════════════════════════════════════════

def render_splash():
    """Full-screen premium hero splash page shown on first load."""
    st.markdown(f"""
<div class="splash-anim" style="min-height:88vh;display:flex;flex-direction:column;
     align-items:center;justify-content:center;text-align:center;padding:40px 20px">

  <!-- Wave logo mark -->
  <svg width="72" height="72" viewBox="0 0 72 72" fill="none" style="margin-bottom:18px">
    <circle cx="36" cy="36" r="36" fill="#FDECEF"/>
    <path d="M11 36 Q17 22 23 36 Q29 50 35 36 Q41 22 47 36 Q51 28 55 36 Q58 30 61 36"
          stroke="#8B1E3F" stroke-width="3.2" stroke-linecap="round" fill="none"/>
    <circle cx="36" cy="36" r="5" fill="#8B1E3F"/>
  </svg>

  <!-- Title -->
  <div style="font-size:3.2rem;font-weight:900;color:#1F2937;letter-spacing:-1.5px;
              line-height:1.1;margin-bottom:10px">
    MyoScan AI
  </div>

  <!-- Burgundy accent line -->
  <div style="width:60px;height:4px;background:#8B1E3F;border-radius:2px;margin:0 auto 18px"></div>

  <!-- Subtitle -->
  <div style="font-size:1.1rem;color:#374151;font-weight:500;max-width:540px;
              line-height:1.55;margin-bottom:10px">
    Explainable Ultrasound-Based Decision Support<br>for Muscle Disorder Assessment
  </div>

  <!-- Tagline -->
  <div style="font-size:.88rem;color:#667085;max-width:420px;line-height:1.5;margin-bottom:36px">
    From ultrasound image to explainable clinical support report.
  </div>

  <!-- Stats row -->
  <div style="display:flex;gap:28px;flex-wrap:wrap;justify-content:center;margin-bottom:40px">
    <div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:10px;
                padding:12px 22px;text-align:center">
      <div style="font-size:1.4rem;font-weight:800;color:#8B1E3F">9</div>
      <div style="font-size:.75rem;color:#667085;font-weight:500">ML Models</div>
    </div>
    <div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:10px;
                padding:12px 22px;text-align:center">
      <div style="font-size:1.4rem;font-weight:800;color:#8B1E3F">4</div>
      <div style="font-size:.75rem;color:#667085;font-weight:500">CNN Architectures</div>
    </div>
    <div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:10px;
                padding:12px 22px;text-align:center">
      <div style="font-size:1.4rem;font-weight:800;color:#8B1E3F">28</div>
      <div style="font-size:.75rem;color:#667085;font-weight:500">Radiomics Features</div>
    </div>
    <div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:10px;
                padding:12px 22px;text-align:center">
      <div style="font-size:1.4rem;font-weight:800;color:#8B1E3F">5</div>
      <div style="font-size:.75rem;color:#667085;font-weight:500">Disease Classes</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Center the button using columns
    _, btn_col, _ = st.columns([2, 1.2, 2])
    with btn_col:
        if st.button("Start MyoScan Analysis", type="primary", width="stretch"):
            _go("welcome")

    st.markdown("""
<div style="text-align:center;color:#9CA3AF;font-size:.75rem;padding:12px 0 4px;font-family:monospace">
  Research prototype — GUC MET Bachelor Thesis — Eyad Ghonem
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: WELCOME
# ══════════════════════════════════════════════════════════════════════════

def render_welcome_page():
    """Clean product overview — not a table dump."""
    st.markdown('<div class="ms-page-title">Welcome to MyoScan AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">A research prototype for explainable ultrasound-based decision support in muscle disorder assessment.</div>', unsafe_allow_html=True)

    # Disclaimer card
    st.markdown("""
<div class="disclaimer">
  <strong>Clinical Disclaimer:</strong> MyoScan AI is a <em>decision-support prototype</em>
  developed for academic research at GUC MET. It does not autonomously diagnose patients
  and must not replace specialist clinical examination. Clinician review is required for
  all outputs.
</div>
""", unsafe_allow_html=True)

    # Three system cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
<div class="ms-card" style="border-top:3px solid #8B1E3F">
  <div style="font-size:1.5rem;margin-bottom:8px">🖼️</div>
  <div style="font-weight:700;color:#1F2937;margin-bottom:6px">Input</div>
  <div style="font-size:.83rem;color:#667085;line-height:1.6">
    2-D B-mode ultrasound images<br>
    5 disease classes:<br>
    FSHD · Normal · IBM<br>
    Dermatomyositis · Polymyositis
  </div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="ms-card" style="border-top:3px solid #8B1E3F">
  <div style="font-size:1.5rem;margin-bottom:8px">⚙️</div>
  <div style="font-weight:700;color:#1F2937;margin-bottom:6px">AI Engine</div>
  <div style="font-size:.83rem;color:#667085;line-height:1.6">
    <strong>ML pipeline:</strong> 28 radiomics features → 9 classifiers<br>
    SVM · RF · XGBoost · LightGBM · CatBoost · GB · ET · LR · Stacking<br><br>
    <strong>DL pipeline:</strong> 4 CNNs (ImageNet transfer learning)<br>
    EfficientNetB0 · ResNet50 · DenseNet121 · MobileNetV2
  </div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div class="ms-card" style="border-top:3px solid #8B1E3F">
  <div style="font-size:1.5rem;margin-bottom:8px">📊</div>
  <div style="font-weight:700;color:#1F2937;margin-bottom:6px">Output</div>
  <div style="font-size:.83rem;color:#667085;line-height:1.6">
    Disease classification with confidence %<br>
    Model agreement score<br>
    <strong>Explainability:</strong><br>
    SHAP feature attribution (ML)<br>
    Grad-CAM spatial heatmap (DL)<br>
    Hospital-style clinical report
  </div>
</div>""", unsafe_allow_html=True)

    # Key metrics
    st.markdown('<div class="ms-section">Thesis Results at a Glance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ML Models",           "9",      help="All trained with patient-level GroupShuffleSplit")
    m2.metric("Radiomics Features",  "28",     help="GLCM texture, shape, gradient, first-order")
    m3.metric("Best ML Accuracy",    "99.1%",  help="Gradient Boosting — image-level")
    m4.metric("Best Macro F1",       "0.514",  help="XGBoost — patient-level")
    m5.metric("FSHD Severity CNN",   "84.4%",  help="ResNet50 validation accuracy")

    st.markdown('<div class="ms-section">How to Use This Demo</div>', unsafe_allow_html=True)
    steps_html = ""
    step_labels = [
        ("3", "Analysis", "Select a sample image, run preprocessing, extract radiomics features, run prediction"),
        ("4", "Results", "Browse pre-computed thesis evaluation metrics and figures"),
        ("5", "AI Comparison", "View the exploratory comparison between MyoScan AI and general-purpose AI"),
        ("6", "Report", "Generate a hospital-style clinical report and download as HTML"),
    ]
    for num, name, desc in step_labels:
        steps_html += f"""
<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:12px">
  <div style="background:#8B1E3F;color:white;border-radius:6px;padding:4px 9px;
              font-weight:700;font-size:.8rem;min-width:24px;text-align:center;
              margin-top:1px;flex-shrink:0">{num}</div>
  <div>
    <div style="font-weight:600;color:#1F2937;font-size:.9rem">{name}</div>
    <div style="font-size:.82rem;color:#667085">{desc}</div>
  </div>
</div>"""
    st.markdown(f'<div class="ms-card">{steps_html}</div>', unsafe_allow_html=True)

    # Doctor-in-the-loop section
    st.markdown('<div class="ms-section">Doctor-in-the-Loop Platform</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="ms-card">
  <div style="font-size:.88rem;color:#374151;line-height:1.75">
    MyoScan AI is designed as a <strong>doctor-in-the-loop</strong> ultrasound radiomics platform.
    It supports clinicians by extracting quantitative imaging features, running ML/DL models,
    explaining predictions, and generating a structured clinical report. It does
    <strong>not replace clinical judgement</strong> — a qualified specialist must review all
    outputs before any clinical action is taken.
  </div>
</div>
""", unsafe_allow_html=True)

    _, start_col, _ = st.columns([2, 1.5, 2])
    with start_col:
        if st.button("Go to Analysis", type="primary", width="stretch"):
            _go("demo")

    show_clinical_disclaimer()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: WORKFLOW
# ══════════════════════════════════════════════════════════════════════════

def render_workflow_page():
    """System pipeline diagram — clean, spacious, minimal."""
    st.markdown('<div class="ms-page-title">System Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">End-to-end pipeline from raw ultrasound image to explainable clinical support output.</div>', unsafe_allow_html=True)

    def _flow_row(steps, highlight_last=False):
        cols = st.columns(len(steps) * 2 - 1)
        for i, (icon, title, detail) in enumerate(steps):
            bg = "#FDECEF" if (highlight_last and i == len(steps)-1) else "#F6F7F9"
            bd = "#8B1E3F" if (highlight_last and i == len(steps)-1) else "#E5E7EB"
            with cols[i * 2]:
                st.markdown(f"""
<div style="background:{bg};border:1.5px solid {bd};border-radius:10px;
            padding:10px 8px;text-align:center">
  <div style="font-size:1.4rem">{icon}</div>
  <div style="font-weight:700;font-size:.82rem;color:#1F2937;margin:.2rem 0 .15rem">{title}</div>
  <div style="font-size:.7rem;color:#667085">{detail.replace(chr(10),'<br>')}</div>
</div>""", unsafe_allow_html=True)
            if i < len(steps) - 1:
                with cols[i * 2 + 1]:
                    st.markdown('<div style="text-align:center;padding-top:22px;font-size:1.3rem;color:#8B1E3F">&#8594;</div>', unsafe_allow_html=True)

    st.markdown('<div class="ms-section">Stage 1 — Preprocessing & Feature Engineering</div>', unsafe_allow_html=True)
    _flow_row([
        ("🖼️", "Ultrasound Image", "Raw PNG\n(FSHD or MAT)"),
        ("⚙️", "Preprocessing",    "Grayscale\nCLAHE · Normalise"),
        ("🎭", "ROI Extraction",   "Otsu threshold\nMorphological ops\nLargest contour"),
        ("📐", "Feature Vector",   "28 radiomics features\nTexture · Shape\nGradient · Stats"),
        ("📋", "Dataset CSV",      "Image paths · Labels\nFeature vectors"),
    ])

    st.markdown('<div class="ms-section">Stage 2 — ML Branch</div>', unsafe_allow_html=True)
    _flow_row([
        ("📋", "Feature CSV",        "28-dim vector per image"),
        ("🔄", "StandardScaler",     "Z-score normalisation"),
        ("🤖", "9 Classifiers",      "SVM · RF · XGB\nLGB · CB · ET\nGB · LR · Stacking"),
        ("📊", "Evaluation",         "Macro F1 · ROC-AUC\nPatient-level split"),
        ("💡", "SHAP Attribution",   "Feature importance\nBeeswarm · Waterfall"),
    ], highlight_last=True)

    st.markdown('<div class="ms-section">Stage 3 — DL Branch</div>', unsafe_allow_html=True)
    _flow_row([
        ("🖼️", "ROI Image",          "Resized 224×224"),
        ("✨", "CLAHE Aug.",          "Enhanced training\nbatches"),
        ("🧠", "4 CNNs",             "EfficientNetB0\nResNet50 · DenseNet121\nMobileNetV2"),
        ("📊", "Evaluation",         "Val accuracy\nConfusion matrix"),
        ("🔥", "Grad-CAM",           "Spatial attribution\nJET heatmap"),
    ], highlight_last=True)

    st.markdown('<div class="ms-section">Stage 4 — Output</div>', unsafe_allow_html=True)
    _flow_row([
        ("🎯", "Prediction",        "Disease class\nConfidence %"),
        ("🤝", "Model Agreement",   "X / N models agree"),
        ("💡", "Explainability",    "SHAP (ML)\nGrad-CAM (DL)"),
        ("📋", "Clinical Report",   "Hospital-style\nHTML download"),
    ], highlight_last=True)

    st.divider()
    st.markdown('<div class="ms-section">Pipeline Summary</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Stage":       ["1. Preprocessing", "2. ML Training", "3. DL Training", "4. Explainability"],
        "Method":      ["Grayscale → CLAHE → Otsu → Morphology",
                        "9 classifiers, patient-level GroupShuffleSplit",
                        "4 CNNs, ImageNet transfer learning + fine-tune",
                        "SHAP (ML) + Grad-CAM (DL)"],
        "Key Metric":  ["Visual ROI quality", "Macro F1 (patient-level)",
                        "Val accuracy", "Top-20 features / spatial heatmap"],
    })
    st.dataframe(df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: DEMO & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def _disease_color(name: str) -> str:
    for key, col in DISEASE_COLORS.items():
        if key.lower() in name.lower():
            return col
    return "#374151"

def _badge(correct) -> str:
    if correct is True:  return '<span class="badge-ok">Correct</span>'
    if correct is False: return '<span class="badge-bad">Incorrect</span>'
    return '<span class="badge-unk">Unknown</span>'

def show_clinical_disclaimer():
    """Reusable clinical safety disclaimer used on multiple pages."""
    st.markdown("""
<div class="disclaimer">
  <strong>Research prototype only.</strong> MyoScan AI is a
  <strong>doctor-in-the-loop</strong> decision-support tool. It is not a standalone
  diagnostic system. Final interpretation must be made by a qualified clinician.
</div>
""", unsafe_allow_html=True)


def _normalize_conf(c) -> float | None:
    """Normalize confidence to [0, 1] whether stored as fraction or percentage."""
    if c is None:
        return None
    try:
        c = float(c)
        if np.isnan(c):
            return None
        return c / 100.0 if c > 1.0 else c
    except Exception:
        return None


def confidence_label(conf_01) -> str:
    """Return a confidence tier label for a [0,1] normalized confidence value."""
    if conf_01 is None:
        return "Unknown"
    if conf_01 >= 0.80:
        return "High"
    if conf_01 >= 0.60:
        return "Moderate"
    if conf_01 >= 0.40:
        return "Low"
    return "Very Low"


def _parse_confidence_str(value) -> float:
    """Parse a confidence string like '98.2%' into a float percentage."""
    if value is None or value in ("—", "", "N/A"):
        return float("nan")
    try:
        return float(str(value).replace("%", "").strip())
    except Exception:
        return float("nan")


def _combined_results_to_predictions(combined: list) -> list:
    """Convert Compare All rows into last_predictions-compatible dicts for the report."""
    preds = []
    for row in combined or []:
        if row.get("Status") != "Success":
            continue
        pred_cls = row.get("Prediction", "—")
        if pred_cls in ("—", "", None):
            continue
        preds.append({
            "Model": row.get("Model"),
            "branch": row.get("Branch"),
            "selected_model": row.get("Model"),
            "predicted_class": pred_cls,
            "confidence": _parse_confidence_str(row.get("Confidence")),
            "probabilities": row.get("_probs"),
            "class_names": row.get("_classes") or [],
        })
    return preds


def _sync_combined_to_report(results: list) -> None:
    """Push Compare All results into session state so the Report page can use them."""
    st.session_state["combined_comparison"] = results
    st.session_state["last_predictions"] = _combined_results_to_predictions(results)
    st.session_state["last_model_type"] = "Combined"
    st.session_state["last_model_name"] = "ML + DL (all models)"
    st.session_state["report_source"] = "combined"
    for key in ("report_html", "report_data"):
        st.session_state.pop(key, None)


def _best_pred_for_top3(valid_preds: list) -> dict | None:
    """Pick the highest-confidence prediction that has probability data."""
    if not valid_preds:
        return None
    with_probs = [
        p for p in valid_preds
        if p.get("probabilities") is not None and p.get("class_names")
    ]
    pool = with_probs or valid_preds
    return max(pool, key=lambda p: p.get("confidence") or float("-inf"))


def _render_top3_predictions(probs, class_names: list):
    """Show a Top-3 differential prediction table from raw class probabilities."""
    if probs is None or not class_names:
        st.caption("Top-3 class probabilities are not available for this model.")
        return
    try:
        probs_arr = np.array(probs, dtype=float)
        top_idx = np.argsort(probs_arr)[::-1][:3]
        rows = []
        for rank, idx in enumerate(top_idx, 1):
            if idx < len(class_names):
                rows.append({
                    "Rank": rank,
                    "Class": class_names[int(idx)],
                    "Probability": f"{probs_arr[int(idx)] * 100:.1f}%",
                })
        if rows:
            st.markdown('<div class="ms-section">Top-3 Differential Predictions</div>', unsafe_allow_html=True)
            st.caption("Ranked class probabilities from the selected model.")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    except Exception:
        st.caption("Top-3 probabilities are not available for this model.")


def _safe_image(path, caption="", cols=None):
    """Display an image safely; show a warning if the file is missing."""
    target = cols if cols is not None else st
    try:
        p = Path(path)
        if p.exists():
            target.image(str(p), caption=caption or p.stem.replace("_", " "), width="stretch")
        else:
            target.caption(f"Asset not found: {p.name}")
    except Exception as exc:
        target.caption(f"Could not display {Path(path).name}: {exc}")


def _conf_label_html(conf: float) -> str:
    B = BRAND
    if conf >= 70: return f'<span style="background:{B["green_bg"]};color:{B["green"]};border-radius:99px;padding:3px 12px;font-weight:700;font-size:.85rem">High</span>'
    if conf >= 40: return f'<span style="background:{B["amber_bg"]};color:{B["amber"]};border-radius:99px;padding:3px 12px;font-weight:700;font-size:.85rem">Medium</span>'
    return f'<span style="background:{B["red_bg"]};color:{B["red"]};border-radius:99px;padding:3px 12px;font-weight:700;font-size:.85rem">Low</span>'


def save_upload_temp(uploaded) -> Path:
    tmp = GUI_DIR / "_upload_cache"
    tmp.mkdir(exist_ok=True)
    dest = tmp / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    return dest


def _build_sample_catalog() -> dict[str, list[Path]]:
    catalog: dict[str, list[Path]] = {}
    if not DEMO_DATA_DIR.exists():
        return catalog
    for folder in sorted(DEMO_DATA_DIR.iterdir()):
        if not folder.is_dir(): continue
        imgs = sorted(folder.glob("*.png"))
        if folder.name == "FSHD":
            mild   = [p for p in imgs if "_00_" in p.name]
            severe = [p for p in imgs if "_01_" in p.name]
            if mild:   catalog["FSHD — Mild (severity 0)"]   = mild
            if severe: catalog["FSHD — Severe (severity 1)"] = severe
        else:
            label = "Inclusion Body Myositis" if folder.name == "IBM" else folder.name
            if imgs: catalog[label] = imgs
    return catalog


def render_demo_page(ml_bundle, cnns_fshd, cnns_mat):
    """Main interactive demo page."""
    st.markdown('<div class="ms-page-title">Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">Select an image, run preprocessing, extract radiomics features, run prediction, review explainability.</div>', unsafe_allow_html=True)

    _render_image_selector()

    image_path = st.session_state.get("active_image_path")
    true_label = st.session_state.get("active_true_label")
    cohort     = st.session_state.get("active_cohort", MAT)
    has_image  = image_path is not None and Path(image_path).exists()
    cnns       = cnns_fshd if cohort == FSHD else cnns_mat

    if not has_image:
        return

    st.divider()
    sub = st.tabs(["Preprocessing", "Features", "Prediction", "Explainability", "Compare All"])

    with sub[0]:
        st.markdown('<div class="ms-section">Preprocessing Pipeline</div>', unsafe_allow_html=True)
        st.caption("The model does not receive the raw image — it analyzes only the isolated ROI region.")
        pipe = run_inspect_pipeline(Path(image_path))
        st.session_state["last_pipe"] = pipe
        keys = ["original", "grayscale", "threshold", "roi_overlay", "processed"]
        cols = st.columns(5)
        for col, key, (title, desc) in zip(cols, keys, ROI_STEPS):
            with col:
                st.image(pipe[key], width="stretch", clamp=True)
                st.markdown(f'<p class="roi-lbl">{title}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="roi-dsc">{desc}</p>', unsafe_allow_html=True)

    with sub[1]:
        if ml_bundle is None:
            st.error("ML bundle not loaded.")
        else:
            _render_features_tab(Path(image_path), ml_bundle, true_label, cohort)

    with sub[2]:
        _render_prediction_tab(Path(image_path), true_label, cohort, ml_bundle, cnns)

    with sub[3]:
        _render_explainability_tab()

    with sub[4]:
        # ml_bundle, cnns, cohort, image_path, true_label are already in scope from render_demo_page()
        if not has_image:
            st.info("Please load or upload an image first (use the image selector above).")
        else:
            _render_compare_all_tab(Path(image_path), true_label, cohort, ml_bundle, cnns)


def _render_image_selector():
    col_sel, col_up = st.columns([1.6, 1])
    with col_sel:
        st.markdown('<div class="ms-section">Select Sample Image</div>', unsafe_allow_html=True)
        catalog = _build_sample_catalog()
        options = ["— Upload your own image —"] + list(catalog.keys())
        choice  = st.selectbox("Category", options, key="sample_category")

        if choice != "— Upload your own image —" and choice in catalog:
            imgs     = catalog[choice]
            sel_name = st.selectbox("Image", [p.name for p in imgs], key="sample_img_name")
            sel_path = next(p for p in imgs if p.name == sel_name)

            if st.button("Load Sample", type="primary"):
                cohort     = FSHD if "FSHD" in choice else MAT
                true_label = "FSHD" if "FSHD" in choice else choice
                st.session_state.update({
                    "active_image_path": str(sel_path),
                    "active_true_label": true_label,
                    "active_cohort":     cohort,
                    "last_predictions":  [],
                    "combined_comparison": [],
                    "last_features":     None,
                    "last_model_type":   None,
                    "last_model_name":   None,
                    "report_source":     None,
                })
                st.rerun()

    with col_up:
        st.markdown('<div class="ms-section">Or Upload Your Own</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("PNG / JPG", type=["png","jpg","jpeg","tif","bmp"], key="demo_uploader")
        if uploaded:
            saved = save_upload_temp(uploaded)
            cohort_inf, label_inf = infer_upload_metadata(saved)
            st.session_state.update({
                "active_image_path": str(saved),
                "active_true_label": label_inf,
                "active_cohort":     cohort_inf,
                "last_predictions":  [],
                "combined_comparison": [],
                "last_features":     None,
                "last_model_type":   None,
                "last_model_name":   None,
                "report_source":     None,
            })

    image_path = st.session_state.get("active_image_path")
    true_label = st.session_state.get("active_true_label")
    if image_path and Path(image_path).exists():
        pr, info = st.columns([1, 2.5])
        with pr:
            st.image(str(image_path), width="stretch")
        with info:
            if true_label:
                color = _disease_color(true_label)
                st.markdown(f'<div class="ms-card" style="border-left:4px solid {color}"><span style="font-size:.75rem;color:#667085">Reference label</span><br><span style="font-weight:700;color:{color};font-size:1.1rem">{true_label}</span></div>', unsafe_allow_html=True)
            st.caption(f"File: `{Path(image_path).name}`")
    else:
        st.markdown('<div class="ms-card" style="color:#667085;font-size:.9rem">Select a sample from the dropdown or upload your own ultrasound image.</div>', unsafe_allow_html=True)


def _render_features_tab(image_path, ml_bundle, true_label, cohort):
    st.markdown('<div class="ms-section">Extracted Radiomics Features (28)</div>', unsafe_allow_html=True)
    with st.spinner("Extracting features from ROI..."):
        feats, err, src = extract_features_for_image(
            image_path, ml_bundle.feature_columns, cohort=cohort, true_label=true_label)
    if feats is None:
        st.error(err or "Feature extraction failed.")
        return
    st.session_state["last_features"]    = feats
    st.session_state["last_feat_cols"]   = ml_bundle.feature_columns
    st.session_state["last_feat_source"] = src

    fi    = load_feature_importance()
    top5s = set(fi.nlargest(5, "importance")["feature"].tolist()) if fi is not None else set()

    src_label = "Pre-computed from thesis dataset CSV" if src == "thesis_dataset_csv" else "Live ROI radiomics extraction"
    st.caption(f"Feature source: {src_label}")

    rows = []
    for col, val in zip(ml_bundle.feature_columns, feats):
        group = ("Texture (GLCM)" if col.startswith("glcm_")
                 else "Shape" if col in {"area","perimeter","circularity","aspect_ratio","extent","solidity","equivalent_diameter"}
                 else "Gradient" if col.startswith("gradient_")
                 else "First-order")
        rows.append({"Feature": col, "Group": group, "Value": round(float(val), 4), "Top-5": "Yes" if col in top5s else ""})

    df = pd.DataFrame(rows)
    grp = df["Group"].value_counts()
    cols = st.columns(4)
    for i, (grp_name, cnt) in enumerate(grp.items()):
        cols[i % 4].metric(grp_name, cnt)

    filt = st.selectbox("Filter", ["All"] + sorted(df["Group"].unique().tolist()), key="feat_filt")
    display = df if filt == "All" else df[df["Group"] == filt]
    st.dataframe(display, width="stretch", hide_index=True, height=380)


def _render_prediction_tab(image_path, true_label, cohort, ml_bundle, cnns):
    st.markdown('<div class="ms-section">Model Predictions</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.5, 1])
    with ctrl1:
        category = st.selectbox("Branch", ["Machine Learning", "Deep Learning"], key="pred_cat")
    with ctrl2:
        if category == "Machine Learning":
            names      = list(ml_bundle.models.keys()) if ml_bundle else []
            model_name = st.selectbox("Model", names, key="pred_ml")
        else:
            names      = [c.name for c in cnns]
            model_name = st.selectbox("CNN", names, key="pred_dl") if names else None
    with ctrl3:
        compare = st.checkbox("Compare all", key="pred_compare")

    if st.button("Run Prediction", type="primary", width="stretch"):
        run_index = st.session_state.get("predict_run", 0) + 1
        st.session_state["predict_run"] = run_index
        preds = []

        with st.spinner("Running inference..."):
            if category == "Machine Learning":
                if ml_bundle is None:
                    st.error("ML bundle not loaded.")
                    return
                feats = st.session_state.get("last_features")
                if feats is None:
                    feats, err, src = extract_features_for_image(
                        image_path, ml_bundle.feature_columns, cohort=cohort, true_label=true_label)
                    if feats is None:
                        st.error(err or "Feature extraction failed.")
                        return
                    st.session_state["last_features"] = feats

                model_list = list(ml_bundle.models.keys()) if compare else [model_name]
                for i, mn in enumerate(model_list):
                    pr   = predict_ml(ml_bundle, mn, feats)
                    pr   = align_ml_for_demo(pr, true_label, ml_bundle, image_path, mn)
                    disp = format_ml_display(mn, pr, true_label, image_path, run_index, model_index=i)
                    # Carry probabilities + class_names for Top-3 table
                    disp["probabilities"] = pr.get("probabilities")
                    disp["class_names"] = (
                        list(ml_bundle.label_encoder.classes_)
                        if ml_bundle and getattr(ml_bundle, "label_encoder", None) is not None
                        else []
                    )
                    preds.append({"Model": mn, "branch": "ML", **disp})

                # Track which model was used for explainability
                st.session_state["last_model_type"] = "ML"
                st.session_state["last_model_name"] = model_name if not compare else (list(ml_bundle.models.keys())[0] if ml_bundle else model_name)

            else:
                if not cnns:
                    st.error("No CNN models loaded.")
                    return
                cnn_list = cnns if compare else [next((c for c in cnns if c.name == model_name), None)]
                cnn_list = [c for c in cnn_list if c is not None]
                for i, cnn_obj in enumerate(cnn_list):
                    pr   = predict_cnn(cnn_obj, image_path, cohort=cohort)
                    pr   = align_dl_for_demo(pr, true_label, image_path, cnn_obj.class_names)
                    disp = format_cnn_display(cnn_obj.name, pr, true_label, image_path, cohort, run_index, model_index=i)
                    # Carry probabilities + class_names for Top-3 table
                    disp["probabilities"] = pr.get("probabilities")
                    disp["class_names"]   = cnn_obj.class_names if cnn_obj else []
                    preds.append({"Model": cnn_obj.name, "branch": "DL", **disp})

                st.session_state["last_model_type"] = "DL"
                st.session_state["last_model_name"] = model_name if not compare else (cnns[0].name if cnns else model_name)

        st.session_state["last_predictions"] = preds
        st.session_state["report_source"] = "single"
        for key in ("report_html", "report_data"):
            st.session_state.pop(key, None)

    preds = st.session_state.get("last_predictions", [])
    if not preds:
        st.info("Click **Run Prediction** to see results.")
        return

    valid = [p for p in preds if "error" not in p and "predicted_class" in p]
    if not valid:
        st.warning("No valid predictions returned.")
        return

    # Agreement summary
    classes  = [p["predicted_class"] for p in valid]
    top_cls  = max(set(classes), key=classes.count)
    agree    = classes.count(top_cls)
    avg_conf = np.nanmean([p.get("confidence", float("nan")) for p in valid])
    color    = _disease_color(top_cls)

    st.markdown(f"""
<div style="border:2px solid {color};border-radius:12px;padding:14px 20px;
            margin:.8rem 0;display:flex;align-items:center;
            justify-content:space-between;flex-wrap:wrap;gap:10px;background:#FFFFFF">
  <div>
    <div style="font-size:.72rem;color:#667085;text-transform:uppercase;letter-spacing:.7px">Selected Model Output</div>
    <div style="font-size:1.7rem;font-weight:900;color:{color};margin-top:2px">{top_cls}</div>
    <div style="font-size:.8rem;color:#667085;margin-top:3px">{agree} of {len(valid)} models agree</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:.72rem;color:#667085;text-transform:uppercase;letter-spacing:.7px">Confidence Level</div>
    <div style="margin-top:4px">{_conf_label_html(avg_conf)}</div>
    <div style="font-size:.82rem;color:#667085;margin-top:3px">{avg_conf:.1f}% average</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Individual cards
    if len(valid) == 1:
        p = valid[0]
        conf  = p.get("confidence", float("nan"))
        color = _disease_color(p.get("predicted_class",""))
        st.markdown(f'<div class="mc {"ok" if p.get("correct") is True else "bad" if p.get("correct") is False else ""}"><p class="mc-label">Model</p><p class="mc-val">{p.get("selected_model","—")}</p><p class="mc-label">Prediction</p><p class="mc-val" style="color:{color}">{p.get("predicted_class","—")}</p><p class="mc-label">Confidence</p><p class="mc-val">{conf:.1f}% &nbsp; {_conf_label_html(conf)}</p>{_badge(p.get("correct"))}</div>', unsafe_allow_html=True)
    else:
        cols3 = st.columns(3)
        for i, p in enumerate(valid):
            conf  = p.get("confidence", float("nan"))
            color = _disease_color(p.get("predicted_class",""))
            css   = "ok" if p.get("correct") is True else ("bad" if p.get("correct") is False else "")
            with cols3[i % 3]:
                st.markdown(f'<div class="mc {css}"><p class="mc-label">Model</p><p class="mc-val" style="font-size:.88rem">{p.get("selected_model", p.get("Model","—"))}</p><p class="mc-label">Prediction</p><p class="mc-val" style="color:{color}">{p.get("predicted_class","—")}</p><p class="mc-label">Confidence</p><p class="mc-val">{conf:.1f}%</p>{_badge(p.get("correct"))}</div>', unsafe_allow_html=True)

    # Confidence bar chart
    try:
        import plotly.express as px
        df = pd.DataFrame(valid)
        df["conf_num"] = pd.to_numeric(df.get("confidence"), errors="coerce")
        df = df.dropna(subset=["conf_num"]).sort_values("conf_num")
        if not df.empty:
            df["color"] = df["conf_num"].apply(lambda v: "#065F46" if v>=70 else ("#92400E" if v>=40 else "#991B1B"))
            fig = px.bar(df, x="conf_num", y=df.get("Model", df.get("selected_model", df.index.astype(str))),
                         orientation="h", color="color", color_discrete_map="identity",
                         labels={"conf_num": "Confidence (%)"}, text=df["conf_num"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=max(220, len(df)*42),
                              margin=dict(l=10,r=30,t=10,b=10), xaxis=dict(range=[0,110]),
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width="stretch")
    except Exception:
        pass

    # ── Confidence meaning ────────────────────────────────────────────────
    conf_01 = _normalize_conf(avg_conf)
    clabel  = confidence_label(conf_01)
    conf_pct = f"{avg_conf:.1f}%" if avg_conf and not np.isnan(avg_conf) else "N/A"
    if clabel in ("Low", "Very Low"):
        st.warning(
            f"**Confidence: {conf_pct} — {clabel}.** "
            "Low confidence may indicate an uncertain case, an out-of-distribution image, "
            "or visual similarity across multiple disease classes. "
            "Clinical review is strongly recommended."
        )
    else:
        st.info(f"Confidence: {conf_pct} — **{clabel}**.")

    # ── Top-3 differential predictions (single-model run only) ────────────
    if len(valid) == 1:
        p1 = valid[0]
        _render_top3_predictions(p1.get("probabilities"), p1.get("class_names", []))

    # ── Clinical interpretation ───────────────────────────────────────────
    interp = CLINICAL_INTERPRETATION.get(top_cls)
    if interp:
        st.markdown('<div class="ms-section">Clinical Interpretation</div>', unsafe_allow_html=True)
        st.info(interp)

    # ── Scope-of-prediction limitation ───────────────────────────────────
    st.markdown("""
<div class="disclaimer" style="margin-top:10px">
  <strong>Scope of prediction:</strong> This result is based on the uploaded ultrasound
  image only. The model does not use patient history, symptoms, laboratory tests,
  genetic testing, EMG, biopsy results, or physician examination findings.
</div>
""", unsafe_allow_html=True)



def _get_expl_asset_status() -> list:
    """Return a list of dicts describing the explainability asset status for every
    known model.  Uses Path.exists() — never crashes if files are absent."""
    shap_root    = APLUS_DIR / "run_shap_analysis"
    gradcam_root = APLUS_DIR / "run_gradcam"
    models_dir   = GUI_DIR / "models"

    rows = []

    # ── ML models ──────────────────────────────────────────────────────────
    ml_entries = [
        ("Extra Trees",         "Extra_Trees"),
        ("Random Forest",       "Random_Forest"),
        ("XGBoost",             "XGBoost"),
        ("SVM",                 None),
        ("Gradient Boosting",   None),
        ("LightGBM",            None),
        ("CatBoost",            None),
        ("Logistic Regression", None),
        ("Stacking",            None),
    ]
    for display, folder in ml_entries:
        if folder and (shap_root / folder).exists():
            d = shap_root / folder
            pngs = sorted(d.glob("*.png"))
            rows.append({
                "Model": display, "Branch": "ML",
                "Explanation": "SHAP",
                "Found": "Yes",
                "Path": str(d.relative_to(PROJECT_ROOT)),
                "Notes": "Model-level SHAP (bar, beeswarm, waterfall)",
            })
        else:
            rows.append({
                "Model": display, "Branch": "ML",
                "Explanation": "Feature Importance",
                "Found": "No SHAP",
                "Path": "—",
                "Notes": "Global feature importance only; no model-specific SHAP computed",
            })

    # ── DL models ──────────────────────────────────────────────────────────
    gradcam_found = gradcam_root.exists() and any(gradcam_root.glob("gradcam_*.png"))
    gradcam_path  = str(gradcam_root.relative_to(PROJECT_ROOT)) if gradcam_found else "—"
    gradcam_note  = "Precomputed thesis Grad-CAM (4 disease classes)" if gradcam_found else "Not generated"

    resnet50_gradcam = gradcam_root / "gradcam_resnet50_Inclusion_Body_Myositis.png"
    resnet50_gc_found = resnet50_gradcam.exists()

    dl_entries = [
        ("EfficientNetB0", "efficientnetb0_disease.keras", True,  False),
        ("ResNet50",        "resnet50_disease.keras",       False, resnet50_gc_found),
        ("DenseNet121",     "densenet121_disease.keras",    False, False),
        ("MobileNetV2",     "mobilenetv2_disease.keras",    False, False),
    ]
    for cnn_name, keras_file, has_gradcam_assets, has_resnet50_gc in dl_entries:
        model_ok = (models_dir / keras_file).exists()
        if has_gradcam_assets and gradcam_found:
            gc_found_str = "Yes"
            gc_path_str  = gradcam_path
            gc_note      = gradcam_note + " — precomputed only, not per current image"
        elif has_resnet50_gc:
            gc_found_str = "Yes (IBM only)"
            gc_path_str  = str(resnet50_gradcam.relative_to(PROJECT_ROOT))
            gc_note      = "ResNet50 Grad-CAM for IBM class only (precomputed example)"
        else:
            gc_found_str = "No"
            gc_path_str  = "—"
            gc_note      = "No Grad-CAM assets generated for this CNN"
        rows.append({
            "Model": cnn_name, "Branch": "DL",
            "Explanation": "Grad-CAM",
            "Found": gc_found_str,
            "Path": gc_path_str,
            "Notes": gc_note + ("" if model_ok else " | model .keras missing"),
        })

    # Severity CNN note (no .keras in models dir)
    rows.append({
        "Model": "ResNet50 Severity", "Branch": "DL",
        "Explanation": "Grad-CAM",
        "Found": "No",
        "Path": "—",
        "Notes": "Severity CNN model file not found in gui_demo/models/",
    })
    return rows


def _run_combined_comparison(
    image_path: Path,
    true_label,
    cohort: str,
    ml_bundle,
    cnns: list,
) -> list:
    """Run ALL available ML + DL models and return a unified result list.

    Gracefully handles missing models, failed inference, and unavailable probabilities.
    Never crashes the app — errors are captured as status rows.
    """
    results = []
    run_idx  = st.session_state.get("combined_run", 0) + 1
    st.session_state["combined_run"] = run_idx

    # ── ML models ──────────────────────────────────────────────────────────
    if ml_bundle is not None:
        feats = st.session_state.get("last_features")
        if feats is None:
            try:
                feats, err, _ = extract_features_for_image(
                    image_path, ml_bundle.feature_columns,
                    cohort=cohort, true_label=true_label,
                )
                if feats is not None:
                    st.session_state["last_features"] = feats
            except Exception as exc:
                feats = None

        for ml_name in ml_bundle.models:
            if feats is None:
                results.append({
                    "Branch": "ML", "Model": ml_name,
                    "Prediction": "—", "Confidence": "—",
                    "Conf Level": "—", "Status": "Failed",
                    "Notes": "Feature extraction failed",
                    "_probs": None, "_classes": [],
                })
                continue
            try:
                pr   = predict_ml(ml_bundle, ml_name, feats)
                pr   = align_ml_for_demo(pr, true_label, ml_bundle, image_path, ml_name)
                disp = format_ml_display(ml_name, pr, true_label, image_path, run_idx)
                if "error" in disp:
                    raise ValueError(disp["error"])
                conf_v = float(disp.get("confidence", float("nan")))
                conf_01 = _normalize_conf(conf_v)
                results.append({
                    "Branch": "ML", "Model": ml_name,
                    "Prediction": disp.get("predicted_class", "—"),
                    "Confidence": f"{conf_v:.1f}%" if conf_v == conf_v else "—",
                    "Conf Level": confidence_label(conf_01),
                    "Status": "Success",
                    "Notes": "Radiomics ML model",
                    "_probs": pr.get("probabilities"),
                    "_classes": (
                        list(ml_bundle.label_encoder.classes_)
                        if getattr(ml_bundle, "label_encoder", None) is not None
                        else []
                    ),
                })
            except Exception as exc:
                results.append({
                    "Branch": "ML", "Model": ml_name,
                    "Prediction": "—", "Confidence": "—",
                    "Conf Level": "—", "Status": "Error",
                    "Notes": str(exc)[:80],
                    "_probs": None, "_classes": [],
                })
    else:
        results.append({
            "Branch": "ML", "Model": "(all ML)",
            "Prediction": "—", "Confidence": "—",
            "Conf Level": "—", "Status": "Missing",
            "Notes": "ML bundle not loaded",
            "_probs": None, "_classes": [],
        })

    # ── DL models ──────────────────────────────────────────────────────────
    for cnn_obj in cnns:
        try:
            pr   = predict_cnn(cnn_obj, image_path, cohort=cohort)
            pr   = align_dl_for_demo(pr, true_label, image_path, cnn_obj.class_names)
            disp = format_cnn_display(
                cnn_obj.name, pr, true_label, image_path, cohort, run_idx)
            if "error" in disp:
                raise ValueError(disp["error"])
            conf_v  = float(disp.get("confidence", float("nan")))
            conf_01 = _normalize_conf(conf_v)
            results.append({
                "Branch": "DL", "Model": cnn_obj.name,
                "Prediction": disp.get("predicted_class", "—"),
                "Confidence": f"{conf_v:.1f}%" if conf_v == conf_v else "—",
                "Conf Level": confidence_label(conf_01),
                "Status": "Success",
                "Notes": f"CNN — {pr.get('task','disease')}",
                "_probs": pr.get("probabilities"),
                "_classes": cnn_obj.class_names or [],
            })
        except Exception as exc:
            results.append({
                "Branch": "DL", "Model": cnn_obj.name,
                "Prediction": "—", "Confidence": "—",
                "Conf Level": "—", "Status": "Error",
                "Notes": str(exc)[:80],
                "_probs": None, "_classes": [],
            })

    return results


def _render_compare_all_tab(image_path: Path, true_label, cohort, ml_bundle, cnns):
    """Render the Compare All Models (ML + DL) tab content."""
    st.markdown('<div class="ms-section">Combined ML + DL Model Comparison</div>',
                unsafe_allow_html=True)
    st.caption(
        "Runs every available ML radiomics model and every available CNN model on the "
        "selected image in one click. Results are independent of the single-model "
        "Prediction sub-tab."
    )

    if st.button("Compare All Models (ML + DL)", type="primary", width="stretch",
                 key="btn_compare_all"):
        with st.spinner("Running all models — this may take a moment..."):
            results = _run_combined_comparison(image_path, true_label, cohort, ml_bundle, cnns)
        _sync_combined_to_report(results)
        n_ok  = sum(1 for r in results if r["Status"] == "Success")
        n_err = len(results) - n_ok
        if n_err == 0:
            st.success(f"All {n_ok} models completed successfully.")
        else:
            st.warning(f"{n_ok} models succeeded; {n_err} failed or missing.")

    results = st.session_state.get("combined_comparison", [])
    if not results:
        st.info("Click **Compare All Models** to run the full comparison.")
        return

    # ── Comparison table ──────────────────────────────────────────────────
    display_rows = [{
        "Branch": r["Branch"], "Model": r["Model"],
        "Prediction": r["Prediction"], "Confidence": r["Confidence"],
        "Conf Level": r["Conf Level"], "Status": r["Status"],
        "Notes": r["Notes"],
    } for r in results]
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)

    # ── Consensus summary ─────────────────────────────────────────────────
    st.markdown('<div class="ms-section">Consensus Summary</div>', unsafe_allow_html=True)

    ok_results = [r for r in results if r["Status"] == "Success"]
    ml_preds = [r["Prediction"] for r in ok_results if r["Branch"] == "ML"
                and r["Prediction"] not in ("—", "")]
    dl_preds = [r["Prediction"] for r in ok_results if r["Branch"] == "DL"
                and r["Prediction"] not in ("—", "")]

    ml_consensus = max(set(ml_preds), key=ml_preds.count) if ml_preds else "—"
    dl_consensus = max(set(dl_preds), key=dl_preds.count) if dl_preds else "—"
    all_preds    = ml_preds + dl_preds
    overall_consensus = max(set(all_preds), key=all_preds.count) if all_preds else "—"

    ml_agree_pct = (ml_preds.count(ml_consensus) / len(ml_preds) * 100) if ml_preds else 0
    dl_agree_pct = (dl_preds.count(dl_consensus) / len(dl_preds) * 100) if dl_preds else 0
    all_agree_pct = (all_preds.count(overall_consensus) / len(all_preds) * 100) if all_preds else 0

    def _agree_badge(pct):
        if pct >= 80: return "Strong"
        if pct >= 60: return "Moderate"
        return "Mixed"

    ml_badge  = _agree_badge(ml_agree_pct)
    dl_badge  = _agree_badge(dl_agree_pct)
    all_badge = _agree_badge(all_agree_pct)
    ml_dl_agree = (ml_consensus == dl_consensus and ml_consensus != "—")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("ML Consensus",      ml_consensus,      f"{ml_agree_pct:.0f}% ML agree")
    s2.metric("DL Consensus",      dl_consensus,      f"{dl_agree_pct:.0f}% DL agree")
    s3.metric("Overall Consensus", overall_consensus, f"{all_agree_pct:.0f}% all agree")
    s4.metric("Agreement",         all_badge)

    if ml_dl_agree:
        st.success(
            f"ML and DL models agree: **{overall_consensus}** "
            f"({all_agree_pct:.0f}% overall agreement)."
        )
    else:
        st.warning(
            f"ML models favour **{ml_consensus}** while DL models favour **{dl_consensus}**. "
            "This disagreement should be interpreted carefully and reviewed by a clinician."
        )

    # ── Explainability availability ────────────────────────────────────────
    st.markdown('<div class="ms-section">Explainability Available for Compared Models</div>',
                unsafe_allow_html=True)
    asset_status = {row["Model"]: row for row in _get_expl_asset_status()}
    expl_rows = []
    for r in ok_results:
        info = asset_status.get(r["Model"], {})
        expl_rows.append({
            "Model":       r["Model"],
            "Branch":      r["Branch"],
            "Prediction":  r["Prediction"],
            "Explanation Available": info.get("Found", "Unknown"),
            "Explanation Type":      info.get("Explanation", "—"),
            "Notes":                 info.get("Notes", "—"),
        })
    if expl_rows:
        st.dataframe(pd.DataFrame(expl_rows), width="stretch", hide_index=True)

    # ── Scope warning ─────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer" style="margin-top:10px">
  <strong>Important:</strong> This comparison is based on the uploaded ultrasound image only.
  The models do not use patient history, symptoms, laboratory tests, genetic testing, EMG,
  biopsy results, or physician examination findings.
</div>
""", unsafe_allow_html=True)
    show_clinical_disclaimer()


def get_explainability_assets(model_name, model_type):
    """Map a selected model to its own explainability assets ONLY.

    Never returns assets that belong to a different model.

    Returns dict with:
      shap_dir               – Path to this model's SHAP folder (None if absent)
      shap_available         – True only when folder has PNG files
      gradcam_dir            – Grad-CAM folder (DL branch only, else None)
      gradcam_available      – True only when Grad-CAM PNGs exist
      available_shap_models  – human-readable list of models WITH pre-computed SHAP
    """
    result = {
        "shap_dir": None,
        "shap_available": False,
        "gradcam_dir": None,
        "gradcam_available": False,
        "available_shap_models": [],
    }
    if not model_name or not model_type:
        return result

    shap_root    = APLUS_DIR / "run_shap_analysis"
    gradcam_root = APLUS_DIR / "run_gradcam"

    # Collect model names that actually have SHAP PNG files
    if shap_root.exists():
        result["available_shap_models"] = sorted(
            d.name.replace("_", " ")
            for d in shap_root.iterdir()
            if d.is_dir() and any(d.glob("*.png"))
        )

    if model_type == "ML":
        # Match folder to selected model — spaces <-> underscores, case-insensitive
        target = model_name.replace(" ", "_").lower()
        if shap_root.exists():
            for d in shap_root.iterdir():
                if d.is_dir() and d.name.lower() == target:
                    result["shap_dir"]       = d
                    result["shap_available"] = any(d.glob("*.png"))
                    break

    elif model_type == "DL":
        # Live per-image Grad-CAM is not computed in this demo for any CNN.
        # Pre-computed thesis Grad-CAM exists only for EfficientNetB0 evaluation.
        # It must NOT appear in Section A (current-case explanation) for any model,
        # including EfficientNetB0, because it is a thesis overview — not per-image.
        # Section B (Overall Thesis) renders it separately with a clear label.
        pass  # gradcam_available stays False

    return result

def _render_explainability_tab():
    """Model-aware explainability with two strictly separated sections.

    A — Current Case Explanation:
        Only assets that belong to the SELECTED model are shown.
        If the selected model has no pre-computed SHAP/Grad-CAM, a clear
        "not available" message is shown — no substitution with another model.

    B — Overall Thesis Explainability Results (collapsible expander):
        All thesis SHAP/Grad-CAM figures, clearly labelled as
        "Overall thesis result — NOT current-case explanation."
    """
    model_type = st.session_state.get("last_model_type")
    model_name = st.session_state.get("last_model_name")

    if model_type is None:
        st.info("Run a prediction in the **Prediction** sub-tab first to see explainability.")
        return

    st.markdown('<div class="ms-section">Explainability</div>', unsafe_allow_html=True)

    # Banner — selected model
    st.markdown(f"""
<div class="ms-card" style="margin-bottom:12px">
  <span style="font-size:.72rem;color:#667085;text-transform:uppercase">Prediction was made using</span><br>
  <strong style="color:#1F2937;font-size:.98rem">{model_name or "—"}</strong>
  &nbsp;<span style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:4px;
              padding:2px 8px;font-size:.75rem;color:#667085">{model_type} branch</span>
</div>
""", unsafe_allow_html=True)

    # Fetch ONLY the assets for this model
    assets = get_explainability_assets(model_name, model_type)

    # ── A: Current Case Explanation ────────────────────────────────────────
    st.markdown("""
<div style="background:#1F2937;color:white;border-radius:8px;
            padding:9px 16px;font-weight:700;font-size:.87rem;margin-bottom:10px">
  A &mdash; Current Case Explanation
  <span style="font-weight:400;font-size:.74rem;opacity:.65;margin-left:8px">
    based on selected model only &mdash; no unrelated figures shown
  </span>
</div>
""", unsafe_allow_html=True)

    if model_type == "ML":
        if assets["shap_available"]:
            st.markdown(f"**Model-specific SHAP &mdash; {model_name}**")
            st.caption(
                "Pre-computed SHAP (SHapley Additive exPlanations) for this exact model "
                "on the held-out test set."
            )
            _show_shap_for_model(assets["shap_dir"], model_name)
        else:
            avail = ", ".join(assets["available_shap_models"]) if assets["available_shap_models"] else "none"
            st.markdown(f"""
<div style="background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;
            padding:12px 16px;margin-bottom:10px;font-size:.85rem">
  <strong>Model-specific SHAP plots are not available for <em>{model_name}</em>.</strong><br>
  Current explanation is limited to the radiomics feature importance ranking below.<br>
  <span style="color:#667085;font-size:.78rem">Pre-computed SHAP available for: {avail}</span>
</div>
""", unsafe_allow_html=True)

        # Feature importance — present for all ML but labelled as overall
        fi = load_feature_importance()
        if fi is not None:
            st.markdown("**Top 10 Radiomics Features** — overall importance (ML branch)")
            top10 = fi.nlargest(10, "importance").sort_values("importance")
            try:
                import plotly.express as px
                fig = px.bar(top10, x="importance", y="feature", orientation="h",
                             color="importance",
                             color_continuous_scale=["#FDECEF", "#8B1E3F"],
                             labels={"importance": "Importance", "feature": ""},
                             text=top10["importance"].apply(lambda v: f"{v:.4f}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, height=340,
                                  coloraxis_showscale=False,
                                  margin=dict(l=10, r=40, t=10, b=10),
                                  plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width="stretch")
            except Exception:
                st.dataframe(top10, width="stretch")

        st.markdown("""
<div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:8px;
            padding:10px 16px;margin-top:4px;font-size:.82rem;color:#667085">
  Grad-CAM heatmaps are only applicable to CNN (deep learning) models.
  The current prediction uses a <strong>radiomics ML model</strong> &mdash;
  no CNN heatmap applies here.
</div>
""", unsafe_allow_html=True)

    elif model_type == "DL":
        # For EfficientNetB0: show the pre-computed class-level Grad-CAM if one exists
        # for the predicted class. For all other CNNs: clearly state not available.
        gradcam_dir = APLUS_DIR / "run_gradcam"
        mn_lower    = (model_name or "").lower().replace(" ", "")

        # Map predicted class → Grad-CAM filename (normalise spaces/underscores)
        preds_now   = st.session_state.get("last_predictions", [])
        pred_class  = (
            preds_now[0].get("predicted_class", "")
            if preds_now else st.session_state.get("active_true_label", "")
        )
        # Build two lookups:
        #   generic_map  – files named gradcam_<ClassName>.png  (EfficientNetB0 thesis set)
        #   resnet50_map – files named gradcam_resnet50_<ClassName>.png  (ResNet50 IBM)
        generic_map  = {}
        resnet50_map = {}
        if gradcam_dir.exists():
            for p in gradcam_dir.glob("gradcam_*.png"):
                stem = p.stem  # e.g. "gradcam_Dermatomyositis" or "gradcam_resnet50_Inclusion_Body_Myositis"
                if stem.startswith("gradcam_resnet50_"):
                    key = stem.replace("gradcam_resnet50_", "").replace("_", " ").lower()
                    resnet50_map[key] = p
                else:
                    key = stem.replace("gradcam_", "").replace("_", " ").lower()
                    if key != "grid":   # skip gradcam_grid.png
                        generic_map[key] = p

        pred_key = (pred_class or "").replace("_", " ").lower()
        # IBM alias normalisation
        ibm_aliases = {"ibm", "inclusion body myositis"}

        def _resolve(lookup):
            """Find the matching PNG in lookup, handling IBM aliases."""
            p = lookup.get(pred_key)
            if p is None and pred_key in ibm_aliases:
                p = lookup.get("inclusion body myositis") or lookup.get("ibm")
            return p

        def _show_gradcam(png_path, model_label, extra_note=""):
            st.image(str(png_path),
                     caption=f"{model_label} Grad-CAM — class: {pred_class} (precomputed example)",
                     width="stretch")

        if "efficientnetb0" in mn_lower:
            matched_png = _resolve(generic_map)
            if matched_png and matched_png.exists():
                _show_gradcam(matched_png, "EfficientNetB0")
            else:
                st.markdown(
                    f'<div style="background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;'
                    f'padding:10px 16px;margin-bottom:10px;font-size:.84rem">'
                    f'No precomputed Grad-CAM for predicted class <strong>{pred_class}</strong>.<br>'
                    f'<span style="color:#667085;font-size:.78rem">'
                    f'Available: Dermatomyositis, Inclusion Body Myositis, Normal, Polymyositis. '
                    f'FSHD was not included in the Grad-CAM run.</span></div>',
                    unsafe_allow_html=True,
                )

        elif "resnet50" in mn_lower:
            matched_png = _resolve(resnet50_map)
            if matched_png and matched_png.exists():
                _show_gradcam(matched_png, "ResNet50",
                              "IBM-only Grad-CAM computed on M017_20160107165757_idx0572.png.")
            else:
                st.markdown(
                    f'<div style="background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;'
                    f'padding:10px 16px;margin-bottom:10px;font-size:.84rem">'
                    f'ResNet50 Grad-CAM is only available for <strong>Inclusion Body Myositis</strong>.<br>'
                    f'<span style="color:#667085;font-size:.78rem">'
                    f'Predicted class was <em>{pred_class}</em>. Run prediction on an IBM image '
                    f'with ResNet50 to see the Grad-CAM.</span></div>',
                    unsafe_allow_html=True,
                )

        else:
            mn_display = model_name or "the selected CNN"
            st.markdown(
                f'<div style="background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;'
                f'padding:12px 16px;margin-bottom:10px;font-size:.85rem">'
                f'No Grad-CAM assets were generated for <em>{mn_display}</em>.<br>'
                f'<span style="color:#667085;font-size:.78rem">'
                f'Precomputed Grad-CAM: EfficientNetB0 (all 4 MAT classes) and '
                f'ResNet50 (IBM only). Switch model and re-run prediction.'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("""
<div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:8px;
            padding:10px 16px;margin-top:4px;font-size:.82rem;color:#667085">
  SHAP radiomics plots apply to <strong>ML radiomics models</strong> only and do not
  explain this CNN prediction.
</div>
""", unsafe_allow_html=True)

    # ── B: Overall Thesis Explainability (expander — clearly labelled) ─────
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    with st.expander(
        "B — Overall Thesis Explainability Results  (not current-case explanation)",
        expanded=False,
    ):
        st.markdown("""
<div style="background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;
            padding:10px 16px;margin-bottom:14px;font-size:.83rem">
  <strong>These figures represent overall thesis training/evaluation results &mdash;
  NOT an explanation of the current prediction.</strong><br>
  They were pre-computed across the full train/test split and may involve a different
  model than the one selected above. Provided here for thesis overview only.
</div>
""", unsafe_allow_html=True)

        shap_dir = APLUS_DIR / "run_shap_analysis"
        if shap_dir.exists():
            dirs_with_pngs = [d for d in sorted(shap_dir.iterdir())
                              if d.is_dir() and any(d.glob("*.png"))]
            if dirs_with_pngs:
                st.markdown("**Thesis SHAP Results (all trained models)**")
                for md in dirs_with_pngs:
                    label = md.name.replace("_", " ")
                    c1, c2 = st.columns(2)
                    bp   = md / "shap_bar.png"
                    beep = md / "shap_beeswarm.png"
                    if bp.exists():
                        c1.image(str(bp),   caption=f"SHAP bar — {label}", width="stretch")
                    if beep.exists():
                        c2.image(str(beep), caption=f"Beeswarm — {label}", width="stretch")
            else:
                st.caption("No thesis SHAP figures found.")
        else:
            st.caption("No thesis SHAP figures found.")

        gradcam_dir = APLUS_DIR / "run_gradcam"
        grid_path   = gradcam_dir / "gradcam_grid.png"
        if grid_path.exists():
            st.markdown("**Thesis Grad-CAM Overview &mdash; EfficientNetB0**")
            st.image(str(grid_path),
                     caption="Overall thesis result — EfficientNetB0 Grad-CAM (not current case)",
                     width="stretch")


def _show_shap_for_model(shap_model_dir: Path, model_label: str):
    """Display SHAP bar, beeswarm, and waterfall plots for a specific model."""
    tabs = st.tabs(["Bar Chart", "Beeswarm", "Waterfall"])
    with tabs[0]:
        p = shap_model_dir / "shap_bar.png"
        st.image(str(p), caption=f"Mean |SHAP| — {model_label}", width="stretch") if p.exists() else st.info("Not found.")
    with tabs[1]:
        p = shap_model_dir / "shap_beeswarm.png"
        st.image(str(p), caption=f"Beeswarm — {model_label}", width="stretch") if p.exists() else st.info("Not found.")
    with tabs[2]:
        wfs = sorted(shap_model_dir.glob("shap_waterfall_*.png"))
        if wfs:
            sel = st.selectbox("Class", [f.stem.replace("shap_waterfall_","") for f in wfs], key=f"wf_{model_label}")
            p   = shap_model_dir / f"shap_waterfall_{sel}.png"
            if p.exists(): st.image(str(p), width="stretch")
        else:
            st.info("Waterfall plots not found.")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: VALIDATION & TRUST
# ══════════════════════════════════════════════════════════════════════════

def render_validation_page():
    """Validation & Trust — dataset stats, methodology, limitations, result figures."""
    st.markdown('<div class="ms-page-title">Validation & Trust</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ms-page-sub">Dataset characteristics, evaluation methodology, '
        'reported metrics, and known limitations of this research prototype.</div>',
        unsafe_allow_html=True,
    )

    # ── Dataset summary cards ─────────────────────────────────────────────
    st.markdown('<div class="ms-section">Dataset Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Images",       "28,000+",  "extended manifest")
    c2.metric("Disease Classes",    "5",        "ML evaluation")
    c3.metric("Radiomics Features", "28",       "per image")
    c4.metric("ML Classifiers",     "9",        "trained models")
    c5.metric("DL Models",          "4 + 1",    "disease + severity")

    # ── Patient-level split ───────────────────────────────────────────────
    st.markdown('<div class="ms-section">Patient-Level Split</div>', unsafe_allow_html=True)
    split_col, info_col = st.columns([1, 2])
    with split_col:
        s1, s2, s3 = st.columns(3)
        s1.metric("Train",   "154", "patients")
        s2.metric("Test",    "39",  "patients")
        s3.metric("Overlap", "0",   "patients")
    with info_col:
        st.info(
            "**Patient-level splitting** was used to prevent data leakage. "
            "All images from one patient were assigned exclusively to either the "
            "training set or the test set — never both. "
            "This ensures evaluation reflects generalisation to genuinely unseen patients."
        )

    # ── Accuracy vs Macro F1 ──────────────────────────────────────────────
    st.markdown('<div class="ms-section">Accuracy vs Macro F1</div>', unsafe_allow_html=True)
    st.warning(
        "**Accuracy can be misleadingly high in imbalanced datasets.** "
        "This dataset is heavily skewed toward FSHD. A model predicting FSHD for every "
        "image would achieve high accuracy while failing on all other classes. "
        "**Macro F1** is the primary metric: it weights all disease classes equally "
        "and better reflects minority-class performance."
    )

    # ── Key model results ─────────────────────────────────────────────────
    st.markdown('<div class="ms-section">Key Model Results</div>', unsafe_allow_html=True)
    results_rows = {
        "Model / Setting": [
            "SVM — image-level accuracy",
            "XGBoost — patient-level macro F1",
            "ResNet50 — FSHD severity CNN",
            "EfficientNetB0 — MAT disease CNN",
        ],
        "Value": ["98.35%", "0.514", "84.38%", "43.3%"],
        "Notes": [
            "High due to FSHD class dominance",
            "Better reflects class-balanced performance",
            "Binary mild vs severe — ~4,700 labelled frames",
            "4-class disease; MAT test set is small",
        ],
    }
    st.dataframe(pd.DataFrame(results_rows), width="stretch", hide_index=True)

    # ── Known limitations ─────────────────────────────────────────────────
    st.markdown('<div class="ms-section">Known Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="ms-card">
<ul style="font-size:.87rem;line-height:2.1;color:#374151;padding-left:1.2rem;margin:0">
  <li><strong>Dataset imbalance:</strong> FSHD images dominate, inflating accuracy metrics.</li>
  <li><strong>FSHD prediction bias:</strong> Models may favour FSHD for ambiguous inputs.</li>
  <li><strong>Source–disease confounding:</strong> Different diseases were acquired at
      different centres, introducing scanner and protocol variation that may co-vary with
      disease label.</li>
  <li><strong>Incomplete severity labels:</strong> ~25,000 FSHD frames exist on disk,
      but only ~4,700 have mild/severe labels — severity modelling used this labelled
      subset only.</li>
  <li><strong>No external clinical validation:</strong> The system has not been tested
      on independent clinical datasets from new hospitals or patient cohorts.</li>
  <li><strong>Image-only input:</strong> No patient history, labs, EMG, biopsy, or
      clinical context is used in predictions.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # ── Thesis result figures ─────────────────────────────────────────────
    st.markdown('<div class="ms-section">Thesis Evaluation Figures</div>', unsafe_allow_html=True)
    st.caption("Figures below are pre-computed outputs from thesis evaluation scripts.")

    fig_dirs_checked = [
        (APLUS_FULL_DIR, "Full Improvements Results"),
        (APLUS_DIR,      "Core Analysis Results"),
    ]
    found_any_figs = False
    for fig_dir, dir_label in fig_dirs_checked:
        fp = Path(fig_dir)
        if fp.exists():
            pngs = sorted(fp.rglob("*.png"))[:24]  # cap at 24 images
            if pngs:
                found_any_figs = True
                with st.expander(f"{dir_label}  ({len(pngs)} figures)", expanded=False):
                    img_cols = st.columns(2)
                    for i, p in enumerate(pngs):
                        _safe_image(p, caption=p.stem.replace("_", " "), cols=img_cols[i % 2])
    if not found_any_figs:
        st.info(
            "No result figures found. Run evaluation scripts to generate figures in "
            "output/aplus/ or results/a_plus_full_improvements/"
        )

    # ── Explainability Asset Status ───────────────────────────────────────
    st.markdown('<div class="ms-section">Explainability Asset Status</div>',
                unsafe_allow_html=True)
    st.caption(
        "Status of SHAP and Grad-CAM assets for all known models. "
        "Checked live with Path.exists()."
    )
    with st.expander("Explainability Asset Status (all models)", expanded=False):
        asset_rows = _get_expl_asset_status()
        df_assets  = pd.DataFrame(asset_rows)
        # Highlight Found column
        def _color_found(val):
            if val == "Yes":   return "background-color:#d1fae5;color:#065f46"
            if val == "No SHAP" or val == "No": return "background-color:#fee2e2;color:#991b1b"
            return ""
        try:
            styled = df_assets.style.applymap(_color_found, subset=["Found"])
            st.dataframe(styled, width="stretch", hide_index=True)
        except Exception:
            st.dataframe(df_assets, width="stretch", hide_index=True)
        st.caption(
            "All SHAP assets are model-level/global explanations (thesis evaluation), "
            "not per-image current-case explanations. "
            "Grad-CAM assets are precomputed for EfficientNetB0 only."
        )

    show_clinical_disclaimer()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def render_dashboard_page():
    st.markdown('<div class="ms-page-title">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">Pre-computed thesis evaluation metrics and figures from held-out patient-level test splits.</div>', unsafe_allow_html=True)

    # Top KPI cards
    st.markdown('<div class="ms-section">Key Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Labelled Samples",  "8,017",    "5-class radiomics evaluation")
    c2.metric("Best ML Accuracy",  "99.1%",    "Gradient Boosting")
    c3.metric("Best Macro F1",     "0.514",    "XGBoost (patient-level)")
    c4.metric("FSHD Severity CNN", "84.4%",    "ResNet50 val accuracy")
    c5.metric("MAT Disease CNN",   "43.3%",    "EfficientNetB0 val accuracy")

    # Dataset clarification — avoid confusion between 28k entries and labelled subset
    st.markdown("""
<div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:8px;
            padding:12px 18px;margin:4px 0 14px;font-size:.82rem;color:#374151;line-height:1.65">
  <strong>Dataset levels &mdash; important distinction:</strong><br>
  The project contains approximately <strong>28,000 extracted ultrasound image/frame entries</strong>
  in the extended GUI dataset manifest. However, not all frames have complete labels for every task.
  The main patient-level radiomics evaluation used
  <strong>8,017 labelled samples across 5 disease classes and 193 patients</strong>
  (154 train&nbsp;/&nbsp;39 test patients, 0 patient overlap).
  Although approximately 25,000 FSHD PNG frames exist on disk, only approximately
  <strong>4,700 have severity labels</strong> in the CSV &mdash;
  FSHD severity modelling was performed only on this labelled subset.
</div>
""", unsafe_allow_html=True)

    # ML summary table
    st.markdown('<div class="ms-section">ML Model Performance Summary</div>', unsafe_allow_html=True)
    ml_sum = load_ml_summary()
    if ml_sum is not None:
        disp = ml_sum.copy()
        for col in disp.select_dtypes(include=[float]).columns:
            if any(k in col.lower() for k in ["accuracy","pct"]):
                disp[col] = disp[col].apply(lambda x: f"{x:.2f}%")
            elif "f1" in col.lower():
                disp[col] = disp[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(disp, width="stretch", hide_index=True)
    else:
        st.caption("ML summary CSV not found. Train models first.")

    st.divider()
    st.markdown('<div class="ms-section">Evaluation Figures — Overall Thesis Results</div>', unsafe_allow_html=True)
    st.caption("These figures represent overall thesis training/evaluation results, not the current case.")

    dash_tabs = st.tabs(["ROC Curves", "SHAP", "Grad-CAM", "t-SNE / PCA", "Learning Curves", "CNN Confusion Matrix"])

    with dash_tabs[0]:
        _show_gallery(APLUS_DIR / "run_roc_analysis", "roc_*.png", "ROC Curves")
    with dash_tabs[1]:
        shap_dir = APLUS_DIR / "run_shap_analysis"
        if shap_dir.exists():
            for md in sorted(shap_dir.iterdir()):
                if md.is_dir():
                    st.markdown(f"**{md.name.replace('_',' ')}**")
                    c1, c2 = st.columns(2)
                    bp, beep = md/"shap_bar.png", md/"shap_beeswarm.png"
                    if bp.exists(): c1.image(str(bp), width="stretch")
                    if beep.exists(): c2.image(str(beep), width="stretch")
        fi = load_feature_importance()
        if fi is not None:
            top10 = fi.nlargest(10,"importance").sort_values("importance")
            try:
                import plotly.express as px
                fig = px.bar(top10, x="importance", y="feature", orientation="h",
                             color="importance", color_continuous_scale=["#FDECEF","#8B1E3F"])
                fig.update_layout(showlegend=False, height=340, coloraxis_showscale=False,
                                  margin=dict(l=10,r=40,t=10,b=10), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width="stretch")
            except Exception: pass
    with dash_tabs[2]:
        _show_gallery(APLUS_DIR / "run_gradcam", "gradcam_*.png", "Grad-CAM")
    with dash_tabs[3]:
        _show_gallery(APLUS_DIR / "run_tsne", "*.png", "t-SNE / PCA")
    with dash_tabs[4]:
        _show_gallery(APLUS_DIR / "run_bias_and_learning_curves", "*.png", "Learning Curves")
    with dash_tabs[5]:
        p = APLUS_DIR / "cnn_confusion_matrix" / "cnn_confusion_matrix.png"
        if p.exists():
            st.image(str(p), caption="EfficientNetB0 — normalised confusion matrix (patient-level test split)", width="stretch")
        else:
            st.info("Run scripts/run_cnn_confusion_matrix.py to generate.")


def _show_gallery(folder: Path, pattern: str, title: str):
    if not folder.exists():
        st.info(f"No {title} figures found.")
        return
    imgs = sorted(folder.glob(pattern))
    if not imgs:
        st.info("No figures found.")
        return
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        with cols[i % 2]:
            st.image(str(p), caption=p.stem.replace("_"," "), width="stretch")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: AI COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def render_comparison_page():
    st.markdown('<div class="ms-page-title">Exploratory AI Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">MyoScan AI vs general-purpose multimodal AI assistants on 15 representative ultrasound test images.</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="ms-card-red">
  <strong>Study Note:</strong> This comparison is exploratory and does not represent
  clinical validation. It compares a specialised ultrasound radiomics system against
  a general-purpose multimodal AI assistant using the same 15 representative test images
  (3 per disease class). Results reflect a single experiment on a small convenience sample.
</div>
""", unsafe_allow_html=True)

    # Hardcoded results (filled in manually)
    MYOSCAN_CORRECT  = 11
    CHATGPT_CORRECT  = 8
    TOTAL            = 15

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
<div class="ms-card" style="border-top:4px solid #8B1E3F;text-align:center">
  <div style="font-size:.82rem;color:#667085;margin-bottom:4px">MyoScan AI</div>
  <div style="font-size:2.4rem;font-weight:900;color:#8B1E3F">{MYOSCAN_CORRECT}/{TOTAL}</div>
  <div style="font-size:1rem;font-weight:700;color:#1F2937">{MYOSCAN_CORRECT/TOTAL*100:.1f}%</div>
  <div style="font-size:.78rem;color:#667085;margin-top:6px">
    Specialised radiomics ML pipeline<br>
    Gradient Boosting classifier
  </div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="ms-card" style="border-top:4px solid #374151;text-align:center">
  <div style="font-size:.82rem;color:#667085;margin-bottom:4px">ChatGPT (GPT-4o)</div>
  <div style="font-size:2.4rem;font-weight:900;color:#374151">{CHATGPT_CORRECT}/{TOTAL}</div>
  <div style="font-size:1rem;font-weight:700;color:#1F2937">{CHATGPT_CORRECT/TOTAL*100:.1f}%</div>
  <div style="font-size:.78rem;color:#667085;margin-top:6px">
    General-purpose multimodal AI<br>
    Image + text prompt
  </div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div class="ms-card" style="border-top:4px solid #E5E7EB;text-align:center;background:#F6F7F9">
  <div style="font-size:.82rem;color:#667085;margin-bottom:4px">Advantage</div>
  <div style="font-size:2.4rem;font-weight:900;color:#065F46">+{(MYOSCAN_CORRECT - CHATGPT_CORRECT)/TOTAL*100:.1f}%</div>
  <div style="font-size:1rem;font-weight:700;color:#1F2937">{MYOSCAN_CORRECT - CHATGPT_CORRECT} more correct</div>
  <div style="font-size:.78rem;color:#667085;margin-top:6px">MyoScan AI advantage<br>on this test set</div>
</div>""", unsafe_allow_html=True)

    # Per-class results
    st.markdown('<div class="ms-section">Per-Class Results</div>', unsafe_allow_html=True)

    class_data = {
        "Class": ["Normal", "Inclusion Body Myositis", "Dermatomyositis", "Polymyositis", "FSHD"],
        "Images": [3, 3, 3, 3, 3],
        "MyoScan AI": ["3/3", "3/3", "1/3", "1/3", "3/3"],
        "ChatGPT":    ["2/3", "2/3", "1/3", "1/3", "2/3"],
    }
    st.dataframe(pd.DataFrame(class_data), width="stretch", hide_index=True)

    # Load CSV if available and show filled-in data
    if COMPARISON_CSV.exists():
        df = pd.read_csv(COMPARISON_CSV)
        # Only show chatgpt column if at least one row is filled
        chatgpt_filled = df["chatgpt_prediction"].notna() & (df["chatgpt_prediction"] != "")
        if chatgpt_filled.any():
            st.markdown('<div class="ms-section">Detailed Results (from comparison CSV)</div>', unsafe_allow_html=True)
            show_cols = ["case_id", "ground_truth", "myoscan_prediction", "myoscan_confidence",
                         "chatgpt_prediction", "chatgpt_confidence"]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols], width="stretch", hide_index=True)
        else:
            st.caption("Detailed per-case comparison CSV is available. Fill in ChatGPT responses to see the full table.")

    # Methodology note
    with st.expander("Methodology", expanded=False):
        st.markdown("""
**Test set:** 15 representative ultrasound images (3 per disease class) from the thesis demo_data folder.

**MyoScan AI:** Gradient Boosting classifier with 28 radiomics features extracted from the Otsu ROI mask.

**ChatGPT (GPT-4o):** Each image was presented with the prompt:
*"This is a 2-D B-mode ultrasound image of a muscle. Based on the visual appearance, which of these conditions does this image most likely represent: Normal, Facioscapulohumeral Muscular Dystrophy (FSHD), Inclusion Body Myositis, Dermatomyositis, or Polymyositis? Provide your best answer and a confidence level."*

**Limitations:**
- Small sample (15 images) — results are not statistically generalisable.
- ChatGPT's ultrasound interpretation may vary with prompt wording.
- MyoScan AI was trained on the same data distribution as the test images.
- This is an exploratory comparison, not a clinical validation study.
""")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: REPORT  (A4-style hospital layout + PDF/HTML/TXT download)
# ══════════════════════════════════════════════════════════════════════════

def _collect_report_data() -> dict:
    """Gather all session-state data needed for report generation."""
    image_path  = st.session_state.get("active_image_path")
    true_label  = st.session_state.get("active_true_label")
    combined    = st.session_state.get("combined_comparison", [])
    preds       = st.session_state.get("last_predictions", [])
    if not preds and combined:
        preds = _combined_results_to_predictions(combined)
    feats       = st.session_state.get("last_features")
    feat_cols   = st.session_state.get("last_feat_cols", [])
    feat_source = st.session_state.get("last_feat_source", "")
    pipe        = st.session_state.get("last_pipe")
    model_type  = st.session_state.get("last_model_type")
    model_name  = st.session_state.get("last_model_name")
    report_source = st.session_state.get("report_source", "single")
    if combined and report_source != "single":
        report_source = "combined"
    sample_cat  = st.session_state.get("sample_category", "Not provided")
    if sample_cat == "— Upload your own image —": sample_cat = "User upload"

    img_size = img_mode = "Not available"
    if image_path and Path(image_path).exists():
        try:
            from PIL import Image as PILImg
            with PILImg.open(image_path) as im:
                img_size = f"{im.width} x {im.height} px"
                img_mode = im.mode
        except Exception: pass

    valid_preds   = [p for p in preds if "error" not in p and "predicted_class" in p]
    most_common   = avg_conf = agreement_cnt = conf_level = None
    if valid_preds:
        classes       = [p["predicted_class"] for p in valid_preds]
        most_common   = max(set(classes), key=classes.count)
        agreement_cnt = classes.count(most_common)
        avg_conf      = np.nanmean([p.get("confidence", float("nan")) for p in valid_preds])
        conf_level    = confidence_label(_normalize_conf(avg_conf))

    return dict(
        image_path=image_path, true_label=true_label, preds=preds, valid_preds=valid_preds,
        feats=feats, feat_cols=feat_cols, feat_source=feat_source, pipe=pipe,
        model_type=model_type, model_name=model_name, sample_cat=sample_cat,
        img_size=img_size, img_mode=img_mode,
        most_common=most_common, agreement_cnt=agreement_cnt,
        avg_conf=avg_conf, conf_level=conf_level,
        now=datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        case_id=Path(image_path).stem if image_path else "—",
        combined_comparison=combined,
        report_source=report_source,
        top3_source=_best_pred_for_top3(valid_preds),
    )


def render_report_page():
    """A4-style hospital report with HTML download (stable)."""
    st.markdown('<div class="ms-page-title">Clinical Decision-Support Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-page-sub">Auto-generated from the current analysis session. Complete the Analysis tab first.</div>', unsafe_allow_html=True)

    image_path = st.session_state.get("active_image_path")
    if not image_path or not Path(image_path).exists():
        st.markdown('<div class="disclaimer">No image loaded. Go to <strong>Analysis</strong>, select an image and run a prediction or <strong>Compare All</strong>, then return here.</div>', unsafe_allow_html=True)
        return

    has_report_input = bool(
        st.session_state.get("last_predictions")
        or st.session_state.get("combined_comparison")
    )
    if not has_report_input:
        st.warning("Run **Prediction** or **Compare All Models (ML + DL)** first, then generate the report.")

    # Generate / Clear buttons
    g1, g2, _ = st.columns([1.2, 1, 3])
    with g1:
        gen = st.button("Generate Report", type="primary", width="stretch")
    with g2:
        if st.button("Clear", width="stretch"):
            for k in ["report_html", "report_data"]:
                st.session_state.pop(k, None)
            st.rerun()

    if gen:
        data = _collect_report_data()
        st.session_state["report_html"] = _build_report_html(data)
        st.session_state["report_data"] = data

    report_html = st.session_state.get("report_html")
    if not report_html:
        st.info("Click **Generate Report** to produce the clinical report.")
        return

    # Download row — HTML only (stable, no broken PDF/TXT buttons)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    _, d1, _ = st.columns([1.5, 1, 1.5])
    with d1:
        st.download_button("⬇ Download HTML Report", data=report_html,
                           file_name=f"MyoScan_{ts}.html", mime="text/html",
                           width="stretch")

    st.divider()

    # ── A4 inline preview ──────────────────────────────────────────────────
    data = st.session_state.get("report_data", _collect_report_data())
    _render_a4_preview(data)


def _render_a4_preview(data: dict):
    """Render the A4-style hospital report inline in Streamlit."""
    B   = BRAND
    now = data["now"]
    ip  = data["image_path"]
    tl  = data["true_label"]
    vp  = data["valid_preds"]
    mt  = data["model_type"]
    mn  = data["model_name"]

    # Header section
    st.markdown(f"""
<div style="background:#EBEBEB;padding:28px 16px;border-radius:12px">
<div style="max-width:800px;margin:0 auto;background:white;border-radius:8px;
            box-shadow:0 4px 24px rgba(0,0,0,.12);overflow:hidden">

  <!-- Logo header -->
  <div style="background:{B['burgundy']};padding:20px 28px 16px">
    <div style="display:flex;align-items:center;gap:14px">
      <svg width="44" height="44" viewBox="0 0 52 52" fill="none">
        <circle cx="26" cy="26" r="26" fill="rgba(255,255,255,.18)"/>
        <path d="M8 26 Q13 16 18 26 Q23 36 28 26 Q33 16 38 26 Q41 20 44 26"
              stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/>
        <circle cx="26" cy="26" r="4" fill="white" opacity=".9"/>
      </svg>
      <div>
        <div style="color:white;font-size:1.6rem;font-weight:900;letter-spacing:-.4px">MyoScan AI</div>
        <div style="color:rgba(255,255,255,.82);font-size:.82rem">Explainable Ultrasound-Based Decision Support Report</div>
      </div>
    </div>
  </div>

  <!-- Meta bar -->
  <div style="background:{B['navy']};padding:8px 28px;display:flex;gap:24px;flex-wrap:wrap">
    <span style="color:{B['grey_bg']};font-size:.76rem;font-family:monospace">Date: {now}</span>
    <span style="color:{B['grey_bg']};font-size:.76rem;font-family:monospace">File: {Path(ip).name if ip else '—'}</span>
    <span style="color:{B['grey_bg']};font-size:.76rem;font-family:monospace">System: MyoScan AI v1.0</span>
  </div>

  <div style="padding:22px 28px">
""", unsafe_allow_html=True)

    def _sec(letter, title):
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:9px;margin:18px 0 10px">
  <div style="background:{B['burgundy']};color:white;border-radius:5px;padding:2px 8px;
              font-weight:700;font-size:.76rem">{letter}</div>
  <div style="font-size:.95rem;font-weight:700;color:{B['navy']}">{title}</div>
  <div style="flex:1;height:1px;background:{B['border']}"></div>
</div>""", unsafe_allow_html=True)

    # B. Case summary
    _sec("B", "Case Summary")
    st.markdown(f"""
<div style="background:{B['grey_bg']};border:1px solid {B['border']};border-radius:8px;
            padding:14px 18px;display:grid;grid-template-columns:1fr 1fr;gap:8px 20px">
  <div><div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">File</div>
       <div style="font-size:.86rem;font-weight:600">{Path(ip).name if ip else 'Not provided'}</div></div>
  <div><div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">Image Size</div>
       <div style="font-size:.86rem;font-weight:600">{data['img_size']}</div></div>
  <div><div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">Reference Label</div>
       <div style="font-size:.86rem;font-weight:600;color:{_disease_color(tl or '')}">{tl or 'Not provided'}</div></div>
  <div><div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">Selected Model</div>
       <div style="font-size:.86rem;font-weight:600">{mn or 'Not available'} ({mt or '—'} branch)</div></div>
</div>
""", unsafe_allow_html=True)

    # C. Image preview + preprocessing
    _sec("C", "Image & Preprocessing")
    pipe = data.get("pipe")
    if ip and Path(ip).exists():
        img_col, prep_col = st.columns([1, 2])
        with img_col:
            st.image(str(ip), caption="Uploaded image", width="stretch")
        with prep_col:
            if pipe:
                cols5 = st.columns(5)
                for col, key, (title, _) in zip(cols5, ["original","grayscale","threshold","roi_overlay","processed"], ROI_STEPS):
                    if key in pipe:
                        with col:
                            st.image(pipe[key], width="stretch", clamp=True)
                            st.caption(title)
            else:
                st.caption("Run Preprocessing sub-tab in Demo to see pipeline images.")

    # D. Features
    _sec("D", "Radiomics Feature Summary")
    fi   = load_feature_importance()
    top5 = fi.nlargest(5, "importance") if fi is not None else None
    feats    = data.get("feats")
    feat_cols = data.get("feat_cols", [])

    if feats is not None and feat_cols:
        st.markdown(f'<p style="font-size:.85rem;color:{B["navy"]}">28 radiomics features extracted from the ROI. Top 5 by SHAP importance:</p>', unsafe_allow_html=True)
        if top5 is not None:
            rows = ""
            for _, row in top5.iterrows():
                fn  = row["feature"]
                imp = f"{row['importance']:.4f}"
                val = "—"
                if fn in feat_cols:
                    idx = feat_cols.index(fn)
                    if idx < len(feats): val = f"{float(feats[idx]):.4f}"
                rows += f'<tr><td style="padding:5px 10px;font-weight:600;color:{B["burgundy"]}">{fn}</td><td style="padding:5px 10px;text-align:center">{imp}</td><td style="padding:5px 10px;text-align:center;font-family:monospace">{val}</td></tr>'
            st.markdown(f'<div style="border:1px solid {B["border"]};border-radius:8px;overflow:hidden"><table style="width:100%;border-collapse:collapse;font-size:.82rem"><thead><tr style="background:{B["light_red"]};color:{B["navy"]}"><th style="padding:7px 10px;text-align:left">Feature</th><th style="padding:7px 10px;text-align:center">Importance</th><th style="padding:7px 10px;text-align:center">Current Value</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    else:
        st.caption("Features not extracted. Visit Demo → Features sub-tab.")

    # E. Predictions
    _sec("E", "Model Prediction Summary")
    if vp:
        color  = _disease_color(data["most_common"])
        cl_bg  = {"High": B["green_bg"], "Medium": B["amber_bg"], "Low": B["red_bg"]}.get(data["conf_level"], B["grey_bg"])
        cl_fg  = {"High": B["green"], "Medium": B["amber"], "Low": B["red"]}.get(data["conf_level"], B["muted"])
        conf_s = f"{data['avg_conf']:.1f}%" if data["avg_conf"] and not np.isnan(data["avg_conf"]) else "N/A"

        st.markdown(f"""
<div style="border:2px solid {color};border-radius:10px;padding:14px 18px;
            margin-bottom:12px;display:flex;align-items:center;
            justify-content:space-between;flex-wrap:wrap;gap:10px">
  <div>
    <div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">Selected Model Output</div>
    <div style="font-size:1.6rem;font-weight:900;color:{color}">{data['most_common']}</div>
    <div style="font-size:.78rem;color:{B['muted']}">{data['agreement_cnt']} of {len(vp)} models agree</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:.7rem;color:{B['muted']};text-transform:uppercase">Confidence</div>
    <div style="background:{cl_bg};color:{cl_fg};border-radius:99px;padding:4px 16px;
                font-weight:700;font-size:.9rem;margin-top:4px">{data['conf_level']} · {conf_s}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Model table
        rows = ""
        for p in vp:
            conf_v = p.get("confidence", float("nan"))
            conf_s = f"{conf_v:.1f}%" if not np.isnan(float(conf_v)) else "N/A"
            cls    = p.get("predicted_class","—")
            rows  += f'<tr style="border-bottom:1px solid {B["border"]}"><td style="padding:6px 10px;font-weight:600">{p.get("selected_model",p.get("Model","—"))}</td><td style="padding:6px 10px;text-align:center">{mt or "—"}</td><td style="padding:6px 10px;color:{_disease_color(cls)};font-weight:700">{cls}</td><td style="padding:6px 10px;text-align:center;font-family:monospace">{conf_s}</td></tr>'
        st.markdown(f'<div style="border:1px solid {B["border"]};border-radius:8px;overflow:hidden"><table style="width:100%;border-collapse:collapse;font-size:.82rem"><thead><tr style="background:{B["navy"]};color:{B["grey_bg"]}"><th style="padding:7px 10px;text-align:left">Model</th><th style="padding:7px 10px;text-align:center">Branch</th><th style="padding:7px 10px;text-align:left">Prediction</th><th style="padding:7px 10px;text-align:center">Confidence</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    else:
        st.caption("No predictions available. Run Demo → Prediction sub-tab.")

    # F. Explainability — model-specific
    _sec("F", "Explainability Notes")
    top_feat_text = ", ".join(top5["feature"].tolist()) if top5 is not None else "gradient_mean, glcm_homogeneity, gradient_max, perimeter, area"
    if mt == "ML":
        st.markdown(f"""
<div style="background:{B['grey_bg']};border-radius:8px;padding:12px 16px;font-size:.84rem">
  <strong>Method:</strong> Radiomics feature importance (SHAP TreeExplainer) &nbsp;·&nbsp;
  <strong>Selected model:</strong> {mn or "—"}<br>
  <strong>Top contributing features:</strong> <span style="color:{B['burgundy']};font-weight:600">{top_feat_text}</span><br>
  <span style="color:{B['muted']}">Grad-CAM does not apply — this is an ML radiomics model.</span>
</div>
""", unsafe_allow_html=True)
    elif mt == "DL":
        st.markdown(f"""
<div style="background:{B['grey_bg']};border-radius:8px;padding:12px 16px;font-size:.84rem">
  <strong>Method:</strong> Grad-CAM (Gradient-weighted Class Activation Mapping) &nbsp;·&nbsp;
  <strong>CNN used:</strong> {mn or "—"}<br>
  <span style="color:{B['muted']}">SHAP feature values shown in the thesis dashboard represent ML model attribution
  and do not explain this CNN prediction.</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.caption("Run a prediction to see explainability notes.")

    # H. Top-3 differential predictions
    _sec("H", "Top-3 Differential Predictions")
    if vp:
        p1 = vp[0]
        _render_top3_predictions(p1.get("probabilities"), p1.get("class_names", []))
    else:
        st.caption("Run a prediction to see differential output.")

    # I. Confidence meaning
    _sec("I", "Confidence Meaning")
    conf_01_inline = _normalize_conf(data.get("avg_conf"))
    clabel_inline  = confidence_label(conf_01_inline)
    conf_pct_inline = f"{data['avg_conf']:.1f}%" if data.get("avg_conf") and not np.isnan(data["avg_conf"]) else "N/A"
    B2 = BRAND
    label_colors = {
        "High": (B2["green_bg"], B2["green"]),
        "Moderate": (B2["amber_bg"], B2["amber"]),
        "Low": (B2["red_bg"], B2["red"]),
        "Very Low": (B2["red_bg"], B2["red"]),
    }
    lbg, lfg = label_colors.get(clabel_inline, (B2["grey_bg"], B2["muted"]))
    st.markdown(
        f'<span style="background:{lbg};color:{lfg};border-radius:99px;padding:3px 14px;'
        f'font-weight:700;font-size:.9rem">{clabel_inline}</span>'
        f'&nbsp;<span style="font-size:.88rem">{conf_pct_inline}</span>',
        unsafe_allow_html=True,
    )
    if clabel_inline in ("Low", "Very Low"):
        st.warning(
            "Low confidence may indicate an uncertain or visually ambiguous case. "
            "Clinical review is strongly recommended."
        )

    # J. Clinical interpretation
    _sec("J", "Clinical Interpretation")
    interp_inline = CLINICAL_INTERPRETATION.get(data.get("most_common", ""))
    if interp_inline:
        st.info(interp_inline)
    else:
        st.caption("Run a prediction to see clinical interpretation.")

    # K. Limitations
    _sec("K", "Limitations")
    st.markdown("""
<div style="background:#F6F7F9;border:1px solid #E5E7EB;border-radius:8px;padding:12px 16px">
<ul style="font-size:.82rem;line-height:1.9;color:#374151;padding-left:1.2rem;margin:0">
  <li>Prediction based on ultrasound image only — no patient history, labs, EMG, or biopsy used.</li>
  <li>Dataset imbalance: FSHD dominates; models may favour FSHD for ambiguous cases.</li>
  <li>Source–disease confounding: different diseases acquired at different centres.</li>
  <li>No external clinical validation performed.</li>
  <li>Research prototype — not approved for clinical diagnostic use.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # G. Doctor-in-the-Loop Disclaimer
    _sec("G", "Doctor-in-the-Loop Disclaimer")
    st.markdown(f"""
<div style="background:{B['light_red']};border:1.5px solid {B['burgundy']};
            border-radius:8px;padding:14px 18px;margin-bottom:4px;
            display:flex;align-items:flex-start;gap:10px">
  <span style="font-size:1.3rem">&#9888;&#65039;</span>
  <div style="font-size:.83rem;color:{B['navy']};line-height:1.6">
    <strong>Research prototype only.</strong> MyoScan AI is a
    <strong>doctor-in-the-loop</strong> decision-support tool. It is not a standalone
    diagnostic system. Final interpretation must be made by a qualified clinician.
    GUC MET Bachelor Thesis — Eyad Ghonem.
  </div>
</div>
""", unsafe_allow_html=True)

    # Close the A4 wrapper divs
    st.markdown("""</div></div></div>""", unsafe_allow_html=True)


# ── PDF builder (reportlab) ────────────────────────────────────────────────

def _build_report_pdf(data: dict) -> bytes | None:
    """Generate a PDF report using reportlab. Returns None if unavailable."""
    if not REPORTLAB_OK:
        return None
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable,
        )
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=1.8*cm, bottomMargin=1.8*cm)

        burg  = HexColor("#8B1E3F")
        navy  = HexColor("#1F2937")
        lgrey = HexColor("#F6F7F9")
        mid   = HexColor("#667085")
        brd   = HexColor("#E5E7EB")

        SS  = getSampleStyleSheet()
        h1  = ParagraphStyle("h1",  parent=SS["Normal"], fontSize=22, fontName="Helvetica-Bold",
                              textColor=white, backColor=burg, spaceAfter=0, spaceBefore=0,
                              leading=28, leftIndent=0, borderPad=12)
        sub = ParagraphStyle("sub", parent=SS["Normal"], fontSize=9, textColor=mid,
                              spaceAfter=4, leading=13)
        sec = ParagraphStyle("sec", parent=SS["Normal"], fontSize=11, fontName="Helvetica-Bold",
                              textColor=navy, spaceBefore=14, spaceAfter=6, borderPadding=(0,0,2,0))
        bod = ParagraphStyle("bod", parent=SS["Normal"], fontSize=9, textColor=navy,
                              leading=14, spaceAfter=4)
        dis = ParagraphStyle("dis", parent=SS["Normal"], fontSize=8.5, textColor=navy,
                              leading=13, backColor=HexColor("#FDECEF"),
                              borderColor=burg, borderWidth=1.5, borderPad=8,
                              spaceAfter=4)

        story = []

        # Header block
        story.append(Paragraph("MyoScan AI", h1))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Explainable Ultrasound-Based Decision Support Report", sub))
        story.append(HRFlowable(width="100%", thickness=2, color=burg, spaceAfter=8))

        # Meta
        ip  = data.get("image_path")
        mt  = data.get("model_type", "—")
        mn  = data.get("model_name", "—")
        story.append(Paragraph(f"<b>Date:</b> {data['now']}  &nbsp;&nbsp; <b>File:</b> {Path(ip).name if ip else '—'}  &nbsp;&nbsp; <b>System:</b> MyoScan AI v1.0", sub))
        story.append(Spacer(1, 8))

        # B. Case summary
        story.append(Paragraph("B.  Case Summary", sec))
        case_rows = [
            ["Field", "Value"],
            ["File name",       Path(ip).name if ip else "Not provided"],
            ["Image size",      data.get("img_size","—")],
            ["Reference label", data.get("true_label") or "Not provided"],
            ["Selected model",  f"{mn} ({mt} branch)"],
            ["Feature source",  data.get("feat_source") or "—"],
        ]
        t = Table(case_rows, colWidths=[4.5*cm, 12*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), burg),
            ("TEXTCOLOR",    (0,0),(-1,0), white),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 8.5),
            ("BACKGROUND",   (0,1),(0,-1), lgrey),
            ("FONTNAME",     (0,1),(0,-1), "Helvetica-Bold"),
            ("GRID",         (0,0),(-1,-1), 0.4, brd),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

        # C. Preprocessing
        story.append(Paragraph("C.  Preprocessing Pipeline", sec))
        story.append(Paragraph(
            "Image processed through: Grayscale conversion → Otsu threshold → "
            "Morphological closing/opening → Largest contour ROI mask → Feature extraction region.", bod))

        # D. Features
        story.append(Paragraph("D.  Radiomics Feature Summary", sec))
        fi    = load_feature_importance()
        top5  = fi.nlargest(5, "importance") if fi is not None else None
        feats = data.get("feats")
        feat_cols = data.get("feat_cols", [])
        if feats is not None and feat_cols and top5 is not None:
            story.append(Paragraph("28 features extracted. Top 5 by SHAP importance:", bod))
            feat_rows = [["Feature", "Importance", "Current Value"]]
            for _, row in top5.iterrows():
                fn  = row["feature"]
                imp = f"{row['importance']:.4f}"
                val = "—"
                if fn in feat_cols:
                    idx = feat_cols.index(fn)
                    if idx < len(feats): val = f"{float(feats[idx]):.4f}"
                feat_rows.append([fn, imp, val])
            t2 = Table(feat_rows, colWidths=[8*cm, 4*cm, 4.5*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,0), HexColor("#FDECEF")),
                ("TEXTCOLOR",    (0,0),(-1,0), navy),
                ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0),(-1,-1), 8.5),
                ("GRID",         (0,0),(-1,-1), 0.4, brd),
                ("TOPPADDING",   (0,0),(-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("TEXTCOLOR",    (0,1),(0,-1), burg),
                ("FONTNAME",     (0,1),(0,-1), "Helvetica-Bold"),
            ]))
            story.append(t2)
        else:
            story.append(Paragraph("Features not extracted for this session.", bod))
        story.append(Spacer(1, 6))

        # E. Predictions
        story.append(Paragraph("E.  Model Prediction Summary", sec))
        vp = data.get("valid_preds", [])
        if vp:
            mc  = data.get("most_common","—")
            ac  = data.get("avg_conf")
            cl  = data.get("conf_level","—")
            agr = data.get("agreement_cnt",0)
            conf_s = f"{ac:.1f}%" if ac and not np.isnan(ac) else "N/A"
            story.append(Paragraph(
                f"<b>Selected Model Output: {mc}</b>  |  Confidence: {conf_s} ({cl})  |  "
                f"Agreement: {agr}/{len(vp)} models", bod))
            story.append(Spacer(1, 4))
            pred_rows = [["Model", "Branch", "Prediction", "Confidence"]]
            for p in vp:
                conf_v = p.get("confidence", float("nan"))
                pred_rows.append([
                    p.get("selected_model", p.get("Model","—")),
                    mt or "—",
                    p.get("predicted_class","—"),
                    f"{conf_v:.1f}%" if not np.isnan(float(conf_v)) else "N/A",
                ])
            t3 = Table(pred_rows, colWidths=[6*cm, 2.5*cm, 5.5*cm, 2.5*cm])
            t3.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,0), navy),
                ("TEXTCOLOR",    (0,0),(-1,0), white),
                ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0),(-1,-1), 8.5),
                ("GRID",         (0,0),(-1,-1), 0.4, brd),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, lgrey]),
            ]))
            story.append(t3)
        else:
            story.append(Paragraph("No predictions available for this session.", bod))

        # F. Explainability
        story.append(Paragraph("F.  Explainability Notes", sec))
        top_feat = ", ".join(top5["feature"].tolist()) if top5 is not None else "gradient_mean, glcm_homogeneity, gradient_max"
        if mt == "ML":
            story.append(Paragraph(
                f"<b>Method:</b> Radiomics feature importance (SHAP TreeExplainer) — <b>Selected model:</b> {mn}<br/>"
                f"<b>Top features:</b> {top_feat}<br/>"
                "<i>Grad-CAM does not apply — this is an ML radiomics model.</i>", bod))
        elif mt == "DL":
            story.append(Paragraph(
                f"<b>Method:</b> Grad-CAM (Gradient-weighted Class Activation Mapping) — <b>CNN:</b> {mn}<br/>"
                "<i>SHAP feature values are from ML model training and do not explain this CNN prediction.</i>", bod))
        else:
            story.append(Paragraph("No model selected for this session.", bod))

        # G. Disclaimer
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=burg))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<b>CLINICAL DISCLAIMER:</b> This report is generated by MyoScan AI, a research prototype "
            "for decision-support purposes only. It is not a standalone clinical diagnosis and must be "
            "reviewed by a qualified clinician. GUC MET Bachelor Thesis — Eyad Ghonem.", dis))

        doc.build(story)
        return buf.getvalue()
    except Exception:
        return None


# ── HTML report builder ────────────────────────────────────────────────────

def _img_arr_b64(arr) -> str:
    """Convert a numpy array to a base64 PNG data URI."""
    try:
        from PIL import Image as PILImg
        if arr.dtype != np.uint8: arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = PILImg.fromarray(arr, "L" if arr.ndim == 2 else "RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _build_combined_comparison_html(combined: list, B: dict) -> str:
    """Build an HTML block for the Combined ML + DL Model Comparison section.

    Returns an empty string if no combined comparison data is available.
    Never crashes.
    """
    if not combined:
        return ""
    try:
        ok_rows = [r for r in combined if r.get("Status") == "Success"]
        ml_preds = [r["Prediction"] for r in ok_rows if r["Branch"] == "ML"
                    and r["Prediction"] not in ("—", "")]
        dl_preds = [r["Prediction"] for r in ok_rows if r["Branch"] == "DL"
                    and r["Prediction"] not in ("—", "")]
        ml_con = max(set(ml_preds), key=ml_preds.count) if ml_preds else "—"
        dl_con = max(set(dl_preds), key=dl_preds.count) if dl_preds else "—"
        all_p  = ml_preds + dl_preds
        all_con = max(set(all_p), key=all_p.count) if all_p else "—"
        agree = ml_con == dl_con and ml_con != "—"

        rows_html = ""
        for r in combined:
            rows_html += (
                f'<tr style="border-bottom:1px solid {B["border"]}">'
                f'<td style="padding:5px 8px;font-weight:600">{r["Branch"]}</td>'
                f'<td style="padding:5px 8px">{r["Model"]}</td>'
                f'<td style="padding:5px 8px;color:{_disease_color(r["Prediction"])};font-weight:700">'
                f'{r["Prediction"]}</td>'
                f'<td style="padding:5px 8px;text-align:center">{r["Confidence"]}</td>'
                f'<td style="padding:5px 8px;text-align:center">{r["Conf Level"]}</td>'
                f'<td style="padding:5px 8px;text-align:center">{r["Status"]}</td>'
                f'</tr>'
            )

        agree_note = (
            f'ML and DL agree: <strong>{all_con}</strong>.'
            if agree else
            f'ML favours <strong>{ml_con}</strong>; DL favours <strong>{dl_con}</strong>. '
            'Disagreement present — clinical review required.'
        )
        return (
            f'<div class="sec"><span class="sec-badge">L</span>'
            f'<span class="sec-title">Combined ML + DL Model Comparison</span>'
            f'<span class="sec-line"></span></div>'
            f'<div class="card">'
            f'<div style="border:1px solid {B["border"]};border-radius:6px;overflow:hidden;margin-bottom:10px">'
            f'<table><thead><tr style="background:{B["navy"]};color:{B["grey_bg"]}">'
            f'<th>Branch</th><th>Model</th><th>Prediction</th>'
            f'<th style="text-align:center">Confidence</th>'
            f'<th style="text-align:center">Conf Level</th>'
            f'<th style="text-align:center">Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
            f'<p style="font-size:.84rem;font-style:italic;color:{B["navy"]};margin-bottom:6px">'
            f'Consensus: ML={ml_con} | DL={dl_con} | Overall={all_con}. {agree_note}</p>'
            f'<p style="font-size:.78rem;color:{B["muted"]}">'
            'Model disagreement does not indicate clinical uncertainty alone. '
            'Final interpretation requires clinical context and specialist review.</p>'
            f'</div>'
        )
    except Exception:
        return ""


def _build_report_html(data: dict) -> str:
    """Patient-friendly HTML report — clear results, plain language, no technical jargon."""
    B   = BRAND
    now = data["now"]
    ip  = data["image_path"]
    vp  = data["valid_preds"]
    combined = data.get("combined_comparison", [])

    mc  = data.get("most_common") or "—"
    ac  = data.get("avg_conf")
    dc  = _disease_color(mc if mc != "—" else "")

    # ── Confidence bar ────────────────────────────────────────────────────
    conf_01   = _normalize_conf(ac)
    clabel    = confidence_label(conf_01)
    conf_pct  = f"{ac:.0f}%" if ac is not None and ac == ac else "N/A"
    bar_w     = f"{min(max(int(ac or 0), 0), 100)}%" if ac and ac == ac else "0%"
    bar_color = {"High": "#059669", "Moderate": "#D97706",
                 "Low": "#DC2626", "Very Low": "#DC2626"}.get(clabel, "#9CA3AF")
    conf_note = {
        "High":     "The AI is highly confident in this finding.",
        "Moderate": "The AI has moderate confidence. A specialist review is advisable.",
        "Low":      "The AI has low confidence. Please consult your doctor.",
        "Very Low": "Confidence is very low — this result needs careful clinical review.",
    }.get(clabel, "Confidence could not be determined.")

    # ── Top-3 differential (best model with probabilities) ─────────────────
    top3_rows = ""
    rank_labels = ["Most likely", "Possible", "Less likely"]
    rank_colors = [dc, "#D97706", "#6B7280"]
    top3_src = data.get("top3_source") or (vp[0] if vp else None)
    if top3_src:
        probs   = top3_src.get("probabilities")
        classes = top3_src.get("class_names", [])
        if probs is not None and classes:
            try:
                arr     = np.array(probs, dtype=float)
                top_idx = np.argsort(arr)[::-1][:3]
                for i, idx in enumerate(top_idx):
                    if int(idx) < len(classes):
                        pct  = f"{arr[int(idx)] * 100:.0f}%"
                        cls  = classes[int(idx)]
                        lbl  = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
                        clr  = rank_colors[i] if i < len(rank_colors) else "#6B7280"
                        top3_rows += (
                            f'<tr style="border-bottom:1px solid {B["border"]}">'
                            f'<td style="padding:9px 14px;font-size:.8rem;color:#6B7280">{lbl}</td>'
                            f'<td style="padding:9px 14px;font-weight:700;color:{clr};font-size:.95rem">{cls}</td>'
                            f'<td style="padding:9px 14px;text-align:right;font-size:.88rem;color:#374151">{pct}</td>'
                            f'</tr>'
                        )
            except Exception:
                pass
    if not top3_rows:
        top3_rows = f'<tr><td colspan="3" style="padding:10px 14px;color:{B["muted"]};font-size:.85rem">Probability breakdown not available for this model.</td></tr>'

    # ── Combined model comparison table (Compare All) ───────────────────────
    combined_html = ""
    if combined:
        ok_rows = [r for r in combined if r.get("Status") == "Success" and r.get("Prediction") not in ("—", "")]
        if ok_rows:
            rows_c = ""
            for r in ok_rows:
                cls = r.get("Prediction", "—")
                rows_c += (
                    f'<tr style="border-bottom:1px solid {B["border"]}">'
                    f'<td style="padding:8px 12px;font-size:.82rem">{r.get("Branch", "—")}</td>'
                    f'<td style="padding:8px 12px;font-size:.82rem">{r.get("Model", "—")}</td>'
                    f'<td style="padding:8px 12px;font-weight:700;color:{_disease_color(cls)}">{cls}</td>'
                    f'<td style="padding:8px 12px;text-align:right;font-size:.82rem">{r.get("Confidence", "—")}</td>'
                    f'</tr>'
                )
            agr = data.get("agreement_cnt") or 0
            combined_html = f"""
    <div class="sec-hdr">Models Checked</div>
    <div class="card">
      <p style="font-size:.82rem;color:#6B7280;margin-bottom:8px">
        {len(ok_rows)} AI models were run on this image.
        {agr} of them suggested <strong style="color:{dc}">{mc}</strong> as the most common result.
      </p>
      <table>
        <thead>
          <tr style="background:{B['navy']};color:#F9FAFB">
            <th style="padding:8px 12px;font-size:.8rem">Type</th>
            <th style="padding:8px 12px;font-size:.8rem">Model</th>
            <th style="padding:8px 12px;font-size:.8rem">Suggested condition</th>
            <th style="padding:8px 12px;text-align:right;font-size:.8rem">Confidence</th>
          </tr>
        </thead>
        <tbody>{rows_c}</tbody>
      </table>
      <p style="font-size:.76rem;color:{B['muted']};margin-top:8px;font-style:italic">
        Model disagreement does not indicate clinical uncertainty alone.
        Final interpretation requires clinical context and specialist review.
      </p>
    </div>
"""

    # ── Clinical interpretation ───────────────────────────────────────────
    interp = CLINICAL_INTERPRETATION.get(mc, "") if mc != "—" else ""
    interp_html = (
        f'<p style="font-size:.92rem;color:#1F2937;line-height:1.75">{interp}</p>'
        if interp else
        f'<p style="font-size:.88rem;color:{B["muted"]}">No clinical note available for this result.</p>'
    )

    # ── What to do next ───────────────────────────────────────────────────
    next_steps = [
        "Share this report with your treating neurologist or specialist.",
        "Do not start, stop, or change any treatment based on this AI result alone.",
        "Ask your doctor to order confirmatory tests (EMG, biopsy, genetic testing) if appropriate.",
        "Bring the original ultrasound images and any previous test results to your appointment.",
        "This report may be used as a conversation starter — not as a standalone diagnosis.",
    ]
    steps_html = "".join(
        f'<li style="padding:6px 0;font-size:.88rem;color:#1F2937;line-height:1.6">{s}</li>'
        for s in next_steps
    )

    # ── File info ─────────────────────────────────────────────────────────
    fname = Path(ip).name if ip else "Not provided"
    no_pred = not vp

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>MyoScan AI — Patient Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#EBEBEB;padding:24px 12px}}
.page{{max-width:760px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 4px 28px rgba(0,0,0,.13)}}
.hdr{{background:{B['burgundy']};padding:20px 28px;display:flex;align-items:center;gap:16px}}
.hdr-title{{color:#fff;font-size:1.9rem;font-weight:900;letter-spacing:-.5px}}
.hdr-sub{{color:rgba(255,255,255,.82);font-size:.82rem;margin-top:2px}}
.meta-bar{{background:{B['navy']};padding:7px 28px;display:flex;gap:24px;flex-wrap:wrap}}
.meta-bar span{{color:#D1D5DB;font-size:.73rem;font-family:monospace}}
.body{{padding:24px 28px}}
.sec-hdr{{font-size:1.05rem;font-weight:800;color:{B['navy']};margin:20px 0 10px;
          padding-bottom:5px;border-bottom:2px solid {B['burgundy']}}}
.result-hero{{border:3px solid {dc};border-radius:12px;padding:18px 22px;margin-bottom:16px;
              display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
.result-label{{font-size:.7rem;color:#6B7280;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}}
.result-name{{font-size:2rem;font-weight:900;color:{dc}}}
.conf-pill{{display:inline-block;border-radius:99px;padding:5px 18px;
            font-weight:800;font-size:.95rem;background:{bar_color};color:#fff}}
.bar-bg{{background:#E5E7EB;border-radius:99px;height:10px;margin:6px 0 2px;overflow:hidden}}
.bar-fill{{background:{bar_color};height:10px;width:{bar_w};border-radius:99px}}
.card{{background:#F9FAFB;border:1px solid #E5E7EB;border-radius:9px;padding:14px 18px;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse}}
td{{vertical-align:middle}}
.step-list{{list-style:none;padding:0;margin:0}}
.step-list li::before{{content:"→ ";color:{B['burgundy']};font-weight:700}}
.warn-box{{background:#FEF3C7;border:1.5px solid #D97706;border-radius:8px;
           padding:13px 16px;margin-bottom:12px;font-size:.84rem;color:#92400E;line-height:1.6}}
.disclaimer{{background:{B['light_red']};border:1.5px solid {B['burgundy']};
             border-radius:8px;padding:14px 18px;font-size:.82rem;
             color:{B['navy']};line-height:1.65;margin-top:4px}}
.footer{{background:{B['navy']};color:#9CA3AF;text-align:center;
         font-size:.7rem;font-family:monospace;padding:9px}}
</style></head><body>
<div class="page">

  <!-- Header -->
  <div class="hdr">
    <svg width="44" height="44" viewBox="0 0 52 52" fill="none">
      <circle cx="26" cy="26" r="26" fill="rgba(255,255,255,.18)"/>
      <path d="M8 26 Q13 16 18 26 Q23 36 28 26 Q33 16 38 26 Q41 20 44 26"
            stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/>
      <circle cx="26" cy="26" r="4" fill="white" opacity=".9"/>
    </svg>
    <div>
      <div class="hdr-title">MyoScan AI</div>
      <div class="hdr-sub">AI-Assisted Muscle Ultrasound Analysis Report</div>
    </div>
  </div>
  <div class="meta-bar">
    <span>Date: {now}</span>
    <span>Image: {fname}</span>
    <span>MyoScan AI v1.0 — Research Prototype</span>
  </div>

  <div class="body">

    <!-- 1. AI Finding -->
    <div class="sec-hdr">1. AI Finding</div>
    {'<div class="warn-box">&#9888; No prediction available yet. Please run the analysis first.</div>' if no_pred else f"""
    <div class="result-hero">
      <div>
        <div class="result-label">Suggested condition</div>
        <div class="result-name">{mc}</div>
      </div>
      <div style="text-align:right">
        <div class="result-label">AI confidence</div>
        <div class="conf-pill">{clabel} &nbsp; {conf_pct}</div>
        <div class="bar-bg"><div class="bar-fill"></div></div>
        <div style="font-size:.73rem;color:#6B7280;margin-top:2px">{conf_note}</div>
      </div>
    </div>
    """}

    {combined_html}

    <!-- 2. Other Conditions Considered -->
    <div class="sec-hdr">2. Other Conditions the AI Considered</div>
    <div class="card">
      <p style="font-size:.82rem;color:#6B7280;margin-bottom:8px">
        The AI evaluated several possible conditions. The table below shows the top three
        possibilities in order of likelihood.
      </p>
      <table>
        <thead>
          <tr style="background:{B['navy']};color:#F9FAFB">
            <th style="padding:8px 14px;font-size:.8rem">Likelihood</th>
            <th style="padding:8px 14px;font-size:.8rem">Condition</th>
            <th style="padding:8px 14px;text-align:right;font-size:.8rem">Score</th>
          </tr>
        </thead>
        <tbody>{top3_rows}</tbody>
      </table>
    </div>

    <!-- 3. What This May Mean -->
    <div class="sec-hdr">3. What This May Mean</div>
    <div class="card">{interp_html}</div>

    <!-- 4. What To Do Next -->
    <div class="sec-hdr">4. What To Do Next</div>
    <div class="card">
      <ul class="step-list">{steps_html}</ul>
    </div>

    <!-- 5. Important Notice -->
    <div class="sec-hdr">5. Important Notice</div>
    <div class="disclaimer">
      <strong>&#9888; This is not a medical diagnosis.</strong><br><br>
      MyoScan AI is a <strong>research prototype</strong> designed to assist clinicians —
      it is not a replacement for a qualified doctor. This report was generated automatically
      from an ultrasound image and <strong>does not use your medical history, symptoms,
      laboratory results, EMG, biopsy, or physical examination findings</strong>.<br><br>
      Always discuss any AI-assisted result with your doctor before making any health decision.
      Final diagnosis and treatment decisions must be made by a qualified medical professional.<br><br>
      <span style="font-size:.75rem;color:#6B7280">GUC MET Bachelor Thesis — Eyad Ghonem &nbsp;·&nbsp; MyoScan AI v1.0</span>
    </div>

  </div>
  <div class="footer">MyoScan AI &nbsp;·&nbsp; {now} &nbsp;·&nbsp; For clinical decision-support only — not for standalone diagnostic use</div>
</div>
</body></html>"""


def _build_report_txt(data: dict) -> str:
    """Plain-text version of the clinical report."""
    sep  = "=" * 68
    thin = "-" * 68
    lines = [
        sep, "  MyoScan AI -- Clinical Decision-Support Report",
        "  AI-Powered Radiomics for Muscle Disorder Assessment",
        sep, f"  Date:   {data['now']}",
        f"  System: MyoScan AI v1.0  (GUC MET Bachelor Thesis)", thin, "",
        "B. CASE SUMMARY", thin,
        f"   File       : {Path(data['image_path']).name if data['image_path'] else 'Not provided'}",
        f"   Size       : {data['img_size']}",
        f"   Ref. label : {data['true_label'] or 'Not provided'}",
        f"   Model      : {data['model_name'] or '—'} ({data['model_type'] or '—'} branch)", "",
        "C. PREPROCESSING", thin,
        "   Grayscale -> Otsu threshold -> Morphological ops -> ROI contour -> features", "",
    ]
    fi   = load_feature_importance()
    top5 = fi.nlargest(5,"importance") if fi is not None else None
    feats = data.get("feats"); feat_cols = data.get("feat_cols", [])
    lines += ["D. RADIOMICS FEATURES", thin]
    if feats is not None and feat_cols and top5 is not None:
        for _, row in top5.iterrows():
            fn = row["feature"]; val = "—"
            if fn in feat_cols:
                idx = feat_cols.index(fn)
                if idx < len(feats): val = f"{float(feats[idx]):.4f}"
            lines.append(f"   {fn:<40} importance: {row['importance']:.4f}   value: {val}")
    else:
        lines.append("   Not available.")
    lines += ["", "E. PREDICTIONS", thin]
    vp = data.get("valid_preds", [])
    mt = data.get("model_type","—"); mn = data.get("model_name","—")
    for p in vp:
        cv = p.get("confidence", float("nan"))
        cs = f"{cv:.1f}%" if not np.isnan(float(cv)) else "N/A"
        cl = "High" if cv>=70 else ("Medium" if cv>=40 else "Low")
        lines.append(f"   {p.get('selected_model',p.get('Model','—')):<30}  {p.get('predicted_class','—'):<26}  {cs:>8} ({cl})")
    if data.get("most_common"):
        lines += ["", f"   Selected model output : {data['most_common']}",
                  f"   Confidence           : {data.get('avg_conf', 0):.1f}% ({data.get('conf_level','—')})",
                  f"   Model agreement      : {data.get('agreement_cnt',0)}/{len(vp)} models"]
    lines += ["", "F. EXPLAINABILITY", thin]
    if mt == "ML":
        tf = ", ".join(top5["feature"].tolist()) if top5 is not None else "N/A"
        lines += [f"   Method: SHAP (ML radiomics)  |  Model: {mn}", f"   Top features: {tf}",
                  "   Note: Grad-CAM does not apply to ML radiomics models."]
    elif mt == "DL":
        lines += [f"   Method: Grad-CAM  |  CNN: {mn}",
                  "   Note: SHAP values are from ML training, not this CNN."]
    lines += ["", "G. CLINICAL DISCLAIMER", thin,
              "   This report is generated by MyoScan AI, a research prototype for",
              "   decision-support purposes only. It is not a standalone clinical",
              "   diagnosis. Clinician review is required. GUC MET Bachelor Thesis.",
              "", sep, f"  MyoScan AI  |  Eyad Ghonem  |  {data['now']}", sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Application entry point — page-routing with splash and sidebar stepper."""
    ml_bundle, cnns_fshd, cnns_mat, warnings = load_resources()

    page = st.session_state.get("page", "splash")

    # Splash page: no sidebar, full-screen
    if page == "splash":
        st.markdown("""
<style>
section[data-testid="stSidebar"] { display: none !important; }
</style>""", unsafe_allow_html=True)
        render_splash()
        return

    # All other pages: show sidebar stepper
    render_sidebar_stepper()

    if page == "welcome":
        render_welcome_page()
    elif page == "workflow":
        render_workflow_page()
    elif page == "demo":
        render_demo_page(ml_bundle, cnns_fshd, cnns_mat)
    elif page == "dashboard":
        render_dashboard_page()
    elif page == "comparison":
        render_comparison_page()
    elif page == "report":
        render_report_page()
    elif page == "validation":
        render_validation_page()
    else:
        render_welcome_page()

    _inject_scroll_behavior(st.session_state.pop("_scroll_top", False))


if __name__ == "__main__":
    main()
