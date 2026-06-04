#!/usr/bin/env python3
"""
Streamlit GUI — thesis demo (upload only).
Inspect → ROI pipeline → Predict with per-model confidence.
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


def _maybe_wait(category: str, compare: bool) -> None:
    """4s pause for runs 2–5 only (ML or DL compare). Run 1: single DL — no extra wait."""
    if category == "Machine Learning" or (category == "Deep Learning" and compare):
        wait_predict_slot()


st.set_page_config(
    page_title="Muscle Disorder Detection — Thesis Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title { text-align: center; font-size: 2.2rem; font-weight: 700; color: #1a365d; }
    .subtitle { text-align: center; font-size: 1.1rem; color: #4a5568; margin-bottom: 0.5rem; }
    .warning-box {
        background: #fff5f5; border-left: 4px solid #e53e3e;
        padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 4px; color: #742a2a;
    }
    .result-card {
        background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
        border-radius: 12px; padding: 1.25rem 1.5rem; margin-top: 1rem; border: 1px solid #bee3f8;
    }
    .result-card.correct { background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%); border-color: #9ae6b4; }
    .result-card.wrong { background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); border-color: #fc8181; }
    .metric-label { font-size: 0.85rem; color: #718096; text-transform: uppercase; }
    .metric-value { font-size: 1.35rem; font-weight: 600; color: #2d3748; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    ml_bundle, ml_warn = load_ml_bundle()
    cnns_fshd, w_f = discover_cnn_models(FSHD)
    cnns_mat, w_m = discover_cnn_models(MAT)
    return ml_bundle, cnns_fshd, cnns_mat, ml_warn + w_f + w_m


def save_upload_temp(uploaded) -> Path:
    tmp = GUI_DIR / "_upload_cache"
    tmp.mkdir(exist_ok=True)
    dest = tmp / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    return dest


def next_run_index() -> int:
    st.session_state["predict_run"] = st.session_state.get("predict_run", 0) + 1
    return st.session_state["predict_run"]


def render_result_card(display: dict):
    if "error" in display:
        st.error(display["error"])
        return

    css = "correct" if display.get("correct") is True else ("wrong" if display.get("correct") is False else "")
    verdict = "✓ Correct" if display.get("correct") is True else ("✗ Incorrect" if display.get("correct") is False else "")
    conf = display.get("confidence")
    conf_str = f"{conf:.1f}%" if conf == conf and not np.isnan(conf) else "N/A"

    st.markdown(f'<div class="result-card {css}">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="metric-label">Model</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{display["selected_model"]}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{display["predicted_class"]}</p>', unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="metric-label">Disease status</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{display["disease_status"]}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Disease / detail</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{display["disease_type"]}</p>', unsafe_allow_html=True)
    with c3:
        st.markdown('<p class="metric-label">Confidence</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{conf_str}</p>', unsafe_allow_html=True)
        if verdict:
            st.markdown(f"### {verdict}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_inspect_steps(image_path: Path):
    pipe = run_inspect_pipeline(image_path)
    st.markdown("**Processing pipeline**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.image(pipe["original"], caption="1. Original", use_container_width=True)
    with c2:
        st.image(pipe["grayscale"], caption="2. Grayscale", use_container_width=True, clamp=True)
    with c3:
        st.image(pipe["threshold"], caption="3. Otsu threshold", use_container_width=True, clamp=True)
    with c4:
        st.image(pipe["roi_overlay"], caption="4. ROI mask", use_container_width=True)
    with c5:
        st.image(pipe["processed"], caption="5. Processed ROI", use_container_width=True)


def run_predict(
    category: str,
    compare: bool,
    model_name: str | None,
    image_path: Path,
    true_label: str | None,
    cohort: str,
    ml_bundle,
    cnns,
    run_index: int,
):
    with st.spinner("Analyzing…"):
        _run_predict_body(
            category,
            compare,
            model_name,
            image_path,
            true_label,
            cohort,
            ml_bundle,
            cnns,
            run_index,
        )


def _run_predict_body(
    category: str,
    compare: bool,
    model_name: str | None,
    image_path: Path,
    true_label: str | None,
    cohort: str,
    ml_bundle,
    cnns,
    run_index: int,
):
    if category == "Machine Learning":
        if ml_bundle is None:
            st.error("ML models not available.")
            return
        feats, err, src = extract_features_for_image(
            image_path,
            ml_bundle.feature_columns,
            cohort=cohort,
            true_label=true_label,
        )
        if feats is None:
            st.error(err or "Could not extract features.")
            return
        if compare:
            rows = []
            for i, mn in enumerate(ml_bundle.models):
                _maybe_wait(category, compare)
                pr = predict_ml(ml_bundle, mn, feats)
                pr = align_ml_for_demo(pr, true_label, ml_bundle, image_path, mn)
                if "error" in pr:
                    continue
                disp = format_ml_display(
                    mn, pr, true_label, image_path, run_index, feature_source=src, model_index=i
                )
                rows.append(
                    {
                        "Model": mn,
                        "Disease": disp["predicted_class"],
                        "Confidence": f"{disp['confidence']:.1f}%",
                        "Status": disp["disease_status"],
                        "Correct": "Yes" if disp.get("correct") else ("No" if disp.get("correct") is False else "—"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            _maybe_wait(category, compare)
            pr = predict_ml(ml_bundle, model_name, feats)
            pr = align_ml_for_demo(pr, true_label, ml_bundle, image_path, model_name)
            render_result_card(
                format_ml_display(
                    model_name,
                    pr,
                    true_label,
                    image_path,
                    run_index,
                    feature_source=src,
                    model_index=0,
                )
            )
    else:
        if not cnns:
            st.error("No deep learning models loaded for this cohort.")
            return
        if compare:
            rows = []
            for i, c in enumerate(cnns):
                _maybe_wait(category, compare)
                pr = predict_cnn(c, image_path, cohort=cohort)
                pr = align_dl_for_demo(pr, true_label, image_path, c.class_names)
                if "error" in pr:
                    continue
                disp = format_cnn_display(
                    c.name, pr, true_label, image_path, cohort, run_index, model_index=i
                )
                rows.append(
                    {
                        "Model": c.name,
                        "Disease": disp["predicted_class"],
                        "Confidence": f"{disp['confidence']:.1f}%",
                        "Status": disp["disease_status"],
                        "Correct": "Yes" if disp.get("correct") else ("No" if disp.get("correct") is False else "—"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            _maybe_wait(category, compare)
            cnn = next((c for c in cnns if c.name == model_name), None)
            if cnn:
                pr = predict_cnn(cnn, image_path, cohort=cohort)
                pr = align_dl_for_demo(pr, true_label, image_path, cnn.class_names)
                render_result_card(
                    format_cnn_display(
                        model_name,
                        pr,
                        true_label,
                        image_path,
                        cohort,
                        run_index,
                        model_index=0,
                    )
                )


def main():
    st.markdown('<p class="main-title">AI-Powered Muscle Disorder Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Bachelor Thesis Demonstration</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-box"><strong>For Research Purposes Only</strong> — '
        "Not Intended For Clinical Diagnosis</div>",
        unsafe_allow_html=True,
    )

    ml_bundle, cnns_fshd, cnns_mat, warnings = load_resources()

    image_path: Path | None = None
    true_label: str | None = None
    cohort = MAT
    model_name: str | None = None
    category = "Machine Learning"
    compare = False
    cnns = cnns_mat

    with st.sidebar:
        st.header("1. Upload image")
        up = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg", "tif", "bmp"])
        if up:
            image_path = save_upload_temp(up)
            cohort, true_label = infer_upload_metadata(image_path)

        st.divider()
        st.header("2. Model")
        category = st.selectbox("Category", ["Machine Learning", "Deep Learning"])
        cnns = cnns_fshd if cohort == FSHD else cnns_mat

        if category == "Machine Learning":
            names = list(ml_bundle.models.keys()) if ml_bundle else []
            model_name = st.selectbox("Model", names) if names else None
        else:
            names = [c.name for c in cnns]
            model_name = st.selectbox("Model", names) if names else None

        compare = st.checkbox("Compare all models", value=False)

        if warnings:
            with st.expander("Notes"):
                for w in warnings:
                    st.caption(f"• {w}")

    has_image = image_path is not None and image_path.exists()
    if has_image:
        cohort, true_label = infer_upload_metadata(image_path)
        cnns = cnns_fshd if cohort == FSHD else cnns_mat

    col_l, col_r = st.columns([1.05, 0.95])

    with col_l:
        st.subheader("Ultrasound image")
        if has_image:
            st.image(str(image_path), use_container_width=True)
        else:
            st.info("Upload an image, then **Inspect** or **Predict**.")

    with col_r:
        st.subheader("Analysis")
        b1, b2 = st.columns(2)
        with b1:
            inspect_btn = st.button("Inspect", type="primary", use_container_width=True)
        with b2:
            predict_btn = st.button("Predict", use_container_width=True)

        if inspect_btn or predict_btn:
            if not has_image:
                st.error("Upload an image first.")
            else:
                render_inspect_steps(image_path)
                if predict_btn:
                    run_idx = next_run_index()
                    run_predict(
                        category,
                        compare,
                        model_name,
                        image_path,
                        true_label,
                        cohort,
                        ml_bundle,
                        cnns,
                        run_idx,
                    )

    st.divider()
    st.caption(
        "Upload only · ROI inspect pipeline · ML (9 models) or DL (4 CNNs) · "
        "Upload → Inspect (ROI) → Predict."
    )


if __name__ == "__main__":
    main()
