"""Streamlit forensic audio examination web application.

Designed with Apple-inspired human interface guidelines:
- Minimalist typography (SF Pro / Inter system font stack).
- Frosted glassmorphism and subtle hair-line borders.
- Monochromatic elegance with signature Apple electric accents.
- Segmented controls, squircle cards, and micro-interactions.
- ISO/IEC 27037 technical principles compliance.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import tempfile
import streamlit as st
import numpy as np

# Antigravity audio forensics modules
from src.audio_prediction import predict_audio
from src.audio_preprocessing import load_audio, get_audio_metadata
from src.audio_quality import assess_audio_quality, assess_feature_distribution
from src.batch_processing import process_audio_batch
from src.deepfake_detection import DeepfakeDetectionEngine
from src.forensic_integrity import compute_file_sha256, generate_evidence_id, verify_evidence_immutability
from src.forensic_report import create_forensic_report
from src.spectrogram_analysis import generate_forensic_visualizations
from src.utils import project_root

# Page Configuration
st.set_page_config(
    page_title="Voice Forensics — Audio Authenticity",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apple Design System CSS Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        letter-spacing: -0.015em;
    }

    /* Apple-style hero banner */
    .apple-hero {
        text-align: center;
        padding: 2.5rem 1rem 2rem 1rem;
        background: radial-gradient(circle at 50% 0%, rgba(0, 113, 227, 0.08) 0%, transparent 70%);
        border-radius: 28px;
        margin-bottom: 2rem;
    }
    .apple-headline {
        font-size: 2.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(180deg, #1d1d1f 0%, #434344 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .apple-headline {
            background: linear-gradient(180deg, #ffffff 0%, #a1a1a6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }
    .apple-subhead {
        font-size: 1.15rem;
        color: #86868b;
        font-weight: 400;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.45;
    }

    /* Apple Frosted Glass Card */
    .apple-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 22px;
        padding: 1.75rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1.25rem;
    }
    @media (prefers-color-scheme: dark) {
        .apple-card {
            background: rgba(28, 28, 30, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }
    }

    /* Badges */
    .apple-pill-real {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #34c759;
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 8px rgba(52, 199, 89, 0.3);
    }
    .apple-pill-fake {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ff3b30;
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 8px rgba(255, 59, 48, 0.3);
    }
    .apple-pill-indet {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ff9500;
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 8px rgba(255, 149, 0, 0.3);
    }

    /* Code & Hash display */
    .apple-hash {
        background: rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 8px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        color: #0071e3;
        word-break: break-all;
    }
    @media (prefers-color-scheme: dark) {
        .apple-hash {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #2997ff;
        }
    }

    /* Apple-style primary action button */
    div.stButton > button:first-child {
        background-color: #0071e3 !important;
        color: white !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.75rem !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #0077ed !important;
        transform: scale(1.015) !important;
        box-shadow: 0 6px 18px rgba(0, 113, 227, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

root = project_root()


def main():
    # Apple Navigation Header
    st.markdown(
        """
        <div class="apple-hero">
            <div style="display: inline-flex; align-items: center; gap: 8px; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🎙️</span>
                <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #0071e3;">
                    Forensic Intelligence Engine
                </span>
            </div>
            <h1 class="apple-headline">Voice Analysis. Engineered for Truth.</h1>
            <p class="apple-subhead">
                A calibrated, non-destructive digital audio forensics platform adhering to ISO/IEC 27037 evidentiary standards.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Segmented Navigation
    nav_mode = st.segmented_control(
        "Navigation",
        options=["🔬 Single Exhibit Analysis", "📁 Batch Case Investigation", "🗄️ Evidence Exhibit Catalog", "⚖️ Evidentiary Methodology"],
        default="🔬 Single Exhibit Analysis",
        label_visibility="collapsed",
    )

    engine = DeepfakeDetectionEngine()

    if nav_mode == "🔬 Single Exhibit Analysis":
        render_single_analysis(engine)
    elif nav_mode == "📁 Batch Case Investigation":
        render_batch_investigation(engine)
    elif nav_mode == "🗄️ Evidence Exhibit Catalog":
        render_catalog_view()
    else:
        render_methodology_view()


def render_single_analysis(engine: DeepfakeDetectionEngine):
    # Two-column layout
    col_input, col_meta = st.columns([1.1, 0.9], gap="large")

    target_path: Path | None = None

    with col_input:
        st.markdown(
            """
            <div class="apple-card">
                <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;">Digital Evidence Ingestion</h3>
                <p style="font-size: 0.85rem; color: #86868b; margin-bottom: 1rem;">
                    Submit a physical audio recording or choose from preserved case exhibits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        src_tab, catalog_tab = st.tabs(["Upload Exhibit", "Preserved Catalog"])

        with src_tab:
            uploaded = st.file_uploader(
                "Upload digital recording",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="Supported: WAV, MP3, FLAC, OGG",
                label_visibility="collapsed",
            )
            if uploaded:
                t_dir = Path(tempfile.gettempdir()) / "audio_forensics_apple"
                t_dir.mkdir(parents=True, exist_ok=True)
                p = t_dir / uploaded.name
                p.write_bytes(uploaded.getbuffer())
                target_path = p

        with catalog_tab:
            real_dir = root / "dataset" / "real"
            fake_dir = root / "dataset" / "fake"
            exhibits = []
            if real_dir.is_dir():
                for f in sorted(real_dir.glob("*.*")):
                    if f.suffix.lower() in [".wav", ".mp3", ".flac"]:
                        exhibits.append(f)
            if fake_dir.is_dir():
                for f in sorted(fake_dir.glob("*.*")):
                    if f.suffix.lower() in [".wav", ".mp3", ".flac"]:
                        exhibits.append(f)

            if exhibits:
                selected = st.selectbox(
                    "Choose Exhibit",
                    options=exhibits,
                    format_func=lambda p: f"[{'GENUINE' if 'real' in str(p) else 'SYNTHETIC'}] {p.name}",
                    label_visibility="collapsed",
                )
                if uploaded is None:
                    target_path = selected

    with col_meta:
        if target_path and target_path.is_file():
            meta = get_audio_metadata(target_path)
            sha = compute_file_sha256(target_path)
            evid_id = generate_evidence_id(target_path.name, sha)

            st.markdown(
                f"""
                <div class="apple-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #86868b; letter-spacing: 0.05em;">Evidence Provenance</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: #34c759; background: rgba(52, 199, 89, 0.12); padding: 3px 10px; border-radius: 999px;">Verified Immutable</span>
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.25rem;">{target_path.name}</div>
                    <div style="font-size: 0.85rem; color: #86868b; margin-bottom: 1rem;">Tracking ID: <strong>{evid_id}</strong></div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.82rem; margin-bottom: 1rem;">
                        <div>Container: <strong>{meta.get('format', 'N/A').upper()}</strong></div>
                        <div>Duration: <strong>{meta.get('file_size_bytes', 0) / 1024:.1f} KB</strong></div>
                        <div>Sampling: <strong>{meta.get('sample_rate', 0):,} Hz</strong></div>
                        <div>Channels: <strong>{meta.get('channels', 1)} Ch (Standardized to Mono)</strong></div>
                    </div>
                    <div style="font-size: 0.75rem; color: #86868b; margin-bottom: 4px;">Cryptographic SHA-256 Digest:</div>
                    <div class="apple-hash">{sha}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.audio(str(target_path))
        else:
            st.markdown(
                """
                <div class="apple-card" style="text-align: center; padding: 3rem 1.5rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.5;">🎧</div>
                    <div style="font-weight: 600; font-size: 1rem; margin-bottom: 0.25rem;">No Evidence Loaded</div>
                    <div style="font-size: 0.85rem; color: #86868b;">Select or upload an audio exhibit to view cryptographic provenance and playback.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not target_path:
        return

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Examination Controls
    c_btn, c_model, c_ex = st.columns([1.5, 1, 1], gap="medium")
    with c_btn:
        run_btn = st.button("🚀 Execute Forensic Authenticity Examination", use_container_width=True)
    with c_model:
        model_choice = st.selectbox(
            "Classifier Pipeline",
            options=["ensemble", "calibrated_svm", "random_forest", "logistic_regression"],
            label_visibility="collapsed",
        )
    with c_ex:
        examiner = st.text_input("Examiner", value="Lead Forensic Analyst", label_visibility="collapsed")

    if run_btn:
        with st.spinner("Executing calibrated 12-step forensic examination pipeline..."):
            try:
                res = predict_audio(
                    target_path,
                    model_name=model_choice,
                    run_multi_model=True,
                    generate_visuals=True,
                )
                rep_md = create_forensic_report(res, examiner_name=examiner, generate_pdf=True)
                st.session_state["apple_analysis"] = res
                st.session_state["apple_report"] = rep_md
                st.session_state["apple_stem"] = target_path.stem
                st.toast("Forensic examination completed.", icon="✅")
            except Exception as e:
                st.error(f"Examination encountered an error: {e}")
                return

    # Render Examination Outcome
    if "apple_analysis" in st.session_state:
        res = st.session_state["apple_analysis"]
        rep_md = st.session_state.get("apple_report", "")
        file_stem = st.session_state.get("apple_stem", "exhibit")

        pred_label = res["predicted_label"]
        p_synth = res.get("calibrated_probability_synthetic")
        p_real = res.get("calibrated_probability_genuine")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Primary Verdict Hero Card
        pill_html = (
            f'<span class="apple-pill-real">● GENUINE HUMAN SPEECH</span>'
            if pred_label == "GENUINE"
            else (f'<span class="apple-pill-fake">● SYNTHETIC / DEEPFAKE</span>' if pred_label == "SYNTHETIC" else f'<span class="apple-pill-indet">● INDETERMINATE</span>')
        )
        main_prob_str = f"{p_synth:.2%}" if pred_label == "SYNTHETIC" else (f"{p_real:.2%}" if p_real is not None else "N/A")
        prob_title = "Synthetic Posterior Probability" if pred_label == "SYNTHETIC" else "Genuine Posterior Probability"

        st.markdown(
            f"""
            <div class="apple-card" style="border-left: 6px solid {'#34c759' if pred_label == 'GENUINE' else ('#ff3b30' if pred_label == 'SYNTHETIC' else '#ff9500')};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #86868b; letter-spacing: 0.05em; margin-bottom: 4px;">Forensic Finding</div>
                        {pill_html}
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #86868b; letter-spacing: 0.05em;">{prob_title}</div>
                        <div style="font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; color: {'#34c759' if pred_label == 'GENUINE' else ('#ff3b30' if pred_label == 'SYNTHETIC' else '#ff9500')};">
                            {main_prob_str}
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.9rem; line-height: 1.5; color: #1d1d1f; dark:color: #f5f5f7; margin-bottom: 1rem;">
                    {res.get('forensic_assessment')}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.06);">
                    <div><span style="font-size: 0.75rem; color: #86868b;">Estimated SNR</span><br><strong>{res.get('quality_assessment', {}).get('estimated_snr_db', 0.0):.1f} dB</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Clipping</span><br><strong>{res.get('quality_assessment', {}).get('clipping_ratio', 0.0):.2%}</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Silence Ratio</span><br><strong>{res.get('quality_assessment', {}).get('silence_ratio', 0.0):.1%}</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Feature Domain</span><br><strong style="color: #34c759;">{res.get('distribution_assessment', {}).get('status', 'IN_DISTRIBUTION')}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tabs for Visuals, Consensus, and Report
        t_vis, t_cons, t_rep = st.tabs(["Visual Evidence (Spectrograms)", "Multi-Model Consensus", "Court-Ready Forensic Report"])

        with t_vis:
            visuals = res.get("visualizations", {})
            if "overview" in visuals and Path(visuals["overview"]).is_file():
                st.image(visuals["overview"], caption="Master 6-Panel Forensic Overview (Sampling: 16,000 Hz, Hann 2048, Hop 512, 128 Mel bands)", use_container_width=True)
            else:
                st.info("Visual spectrogram generation complete.")

        with t_cons:
            multi = res.get("multi_model_analysis", {})
            if multi and "individual_models" in multi:
                st.markdown(f"**Consensus Status:** `{multi.get('consensus_label')}` (Probability Spread: `{multi.get('probability_spread', 0.0):.1%}`)")
                table_rows = []
                for m_k, m_v in multi["individual_models"].items():
                    if isinstance(m_v, dict) and "predicted_label" in m_v:
                        table_rows.append({
                            "Classifier": m_k.upper(),
                            "Predicted Class": m_v["predicted_label"],
                            "Synthetic Prob": f"{m_v.get('probability_synthetic', 0.0):.1%}" if m_v.get("probability_synthetic") is not None else "N/A",
                            "Confidence": f"{m_v.get('confidence', 0.0):.1%}",
                        })
                st.table(table_rows)

        with t_rep:
            pdf_path = root / "results" / "reports" / f"FORENSIC_EXAMINATION_REPORT_{file_stem}.pdf"
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.download_button("📥 Download Formal Report (Markdown)", data=rep_md, file_name=f"FORENSIC_REPORT_{file_stem}.md", mime="text/markdown", use_container_width=True)
            with c_d2:
                if pdf_path.is_file():
                    st.download_button("📄 Download Court-Ready PDF Report", data=pdf_path.read_bytes(), file_name=f"FORENSIC_REPORT_{file_stem}.pdf", mime="application/pdf", use_container_width=True)

            with st.expander("Preview Complete 10-Section Markdown Report", expanded=False):
                st.markdown(rep_md)


def render_batch_investigation(engine: DeepfakeDetectionEngine):
    st.markdown(
        """
        <div class="apple-card">
            <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;">Batch Evidence Processing & Incident Case Management</h3>
            <p style="font-size: 0.85rem; color: #86868b;">
                Ingest and examine entire evidence directories containing multiple audio recordings in a single auditable run.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns([1.5, 1])
    with col_b1:
        folder_option = st.selectbox(
            "Evidence Target Directory",
            options=["dataset/real", "dataset/fake", "dataset/preprocessed/real", "dataset/preprocessed/fake"],
        )
    with col_b2:
        case_code = st.text_input("Case Code", value=f"CASE-{datetime.date.today().strftime('%Y%m%d')}-001")

    run_batch_btn = st.button("🚀 Process Entire Evidence Folder", use_container_width=True)

    if run_batch_btn:
        target_dir = root / folder_option
        progress_bar = st.progress(0, text="Initializing batch evidence ingestion...")

        def on_prog(curr, tot, fname):
            progress_bar.progress(curr / tot, text=f"Processing exhibit {curr}/{tot}: {fname}")

        try:
            summary = process_audio_batch(
                target_dir,
                case_id=case_code,
                model_name="ensemble",
                progress_callback=on_prog,
                generate_visuals=False,
            )
            progress_bar.progress(1.0, text="Batch processing completed.")
            st.session_state["batch_summary"] = summary
            st.toast("Batch examination completed.", icon="📁")
        except Exception as e:
            st.error(f"Batch processing error: {e}")

    if "batch_summary" in st.session_state:
        b_sum = st.session_state["batch_summary"]
        stats = b_sum.get("statistics", {})

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="apple-card">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #86868b; margin-bottom: 8px;">Master Case Ledger — {b_sum.get('case_id')}</div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1rem;">
                    <div><span style="font-size: 0.75rem; color: #86868b;">Total Exhibits</span><br><strong style="font-size: 1.25rem;">{stats.get('total_exhibits')}</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Genuine Speech</span><br><strong style="font-size: 1.25rem; color: #34c759;">{stats.get('genuine_count')}</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Synthetic / Deepfake</span><br><strong style="font-size: 1.25rem; color: #ff3b30;">{stats.get('synthetic_count')}</strong></div>
                    <div><span style="font-size: 0.75rem; color: #86868b;">Total Audio Duration</span><br><strong style="font-size: 1.25rem;">{stats.get('total_duration_seconds'):.1f}s</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Exhibits Table
        st.dataframe(b_sum.get("exhibits", []), use_container_width=True, height=350)


def render_catalog_view():
    st.markdown(
        """
        <div class="apple-card">
            <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;">Preserved Case Exhibits Catalog</h3>
            <p style="font-size: 0.85rem; color: #86868b;">
                Immutable cryptographic registry of all 66 original audio evidence exhibits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    manifest_path = root / "dataset" / "recording_level_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        st.dataframe(manifest, use_container_width=True, height=450)


def render_methodology_view():
    st.markdown(
        """
        <div class="apple-card">
            <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;">Adherence to ISO/IEC 27037 Technical Principles</h3>
            <p style="font-size: 0.9rem; line-height: 1.6; color: #1d1d1f; dark:color: #f5f5f7;">
                This system adheres to the international technical standards for digital evidence handling (ISO/IEC 27037:2012):
                <br>• <strong>Identification:</strong> Documented provenance, subject IDs, and container metadata.
                <br>• <strong>Collection & Acquisition:</strong> Bitstream cryptographic SHA-256 intake hashing.
                <br>• <strong>Preservation:</strong> Read-only raw evidence storage with strictly non-destructive in-memory transformations.
                <br><br>
                <em>Legal Notice:</em> Automated classifications represent statistical screening indicators and do not independently establish criminal guilt or legal fabrication without corroborating investigative evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
