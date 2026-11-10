"""Streamlit web interface for AI Audio Forensics investigation platform.

Phase 4 User Interface providing:
- Evidence file ingestion (Upload or exhibit catalog selector).
- Cryptographic SHA-256 hashing and Evidence ID generation.
- Audio playback and waveform visualization.
- Acoustic quality screening (Duration, Clipping, Silence, SNR).
- Out-of-Distribution (OOD) domain validation.
- Multi-model forensic classification (Calibrated SVM, Random Forest, Logistic Regression).
- Model consensus and disagreement analysis.
- High-resolution spectrograms (STFT, Mel, MFCC heatmaps, Spectral Dynamics).
- Automated 10-section forensic report generation (Markdown & PDF export).
- Audit trail event logging adhering to ISO/IEC 27037 technical principles.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import tempfile
import json
import streamlit as st
import numpy as np

# Antigravity audio forensics modules
from src.audio_prediction import predict_audio
from src.audio_preprocessing import load_audio, get_audio_metadata
from src.audio_quality import assess_audio_quality, assess_feature_distribution
from src.deepfake_detection import DeepfakeDetectionEngine
from src.forensic_integrity import compute_file_sha256, generate_evidence_id, verify_evidence_immutability
from src.forensic_report import create_forensic_report
from src.spectrogram_analysis import generate_forensic_visualizations
from src.utils import project_root

# Page Configuration
st.set_page_config(
    page_title="AI Audio Forensics | Deepfake Voice Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-real {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-fake {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-indet {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .hash-box {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.85rem;
        border-radius: 6px;
        word-break: break-all;
    }
    .callout-card {
        border-left: 4px solid #3B82F6;
        background-color: #EFF6FF;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

root = project_root()


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/audio-wave2.png", width=64)
    st.sidebar.title("Forensic Laboratory")
    st.sidebar.markdown(
        "**AI-Based Voice Analysis for Detecting Synthetic Audio in Criminal Investigations**"
    )
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Navigation",
        [
            "🔬 Audio Authenticity Examination",
            "📊 Exhibit Catalog & Demonstration",
            "📜 Forensic Audit Trail & Chain of Custody",
            "ℹ️ Forensic Methodology & Disclaimers",
        ],
    )

    engine = DeepfakeDetectionEngine()
    available_models = engine.get_available_models()

    st.sidebar.markdown("### Model Status")
    for m in ["calibrated_svm", "random_forest", "logistic_regression"]:
        if m in available_models:
            st.sidebar.markdown(f"✅ `{m}` (Validated)")
        else:
            st.sidebar.markdown(f"⚠️ `{m}` (Unavailable)")

    st.sidebar.markdown("---")
    st.sidebar.caption("Adheres to ISO/IEC 27037 Technical Principles.")

    if mode == "🔬 Audio Authenticity Examination":
        render_examination_page(engine)
    elif mode == "📊 Exhibit Catalog & Demonstration":
        render_catalog_page()
    elif mode == "📜 Forensic Audit Trail & Chain of Custody":
        render_audit_page()
    else:
        render_methodology_page()


def render_examination_page(engine: DeepfakeDetectionEngine):
    st.markdown('<div class="main-header">Audio Authenticity Examination</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated forensic screening for voice synthesis and deepfake audio manipulation.</div>',
        unsafe_allow_html=True,
    )

    # Ingestion Source
    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        upload_tab, catalog_tab = st.tabs(["📁 Upload Audio Exhibit", "🗄️ Select Preserved Case Exhibit"])
        
        target_path: Path | None = None
        is_temp = False

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Submit Digital Audio Evidence",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="WAV, MP3, FLAC, and OGG formats supported.",
            )
            if uploaded_file is not None:
                temp_dir = Path(tempfile.gettempdir()) / "ai_audio_forensics_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                temp_path.write_bytes(uploaded_file.getbuffer())
                target_path = temp_path
                is_temp = True

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
                selected_exhibit = st.selectbox(
                    "Choose Verified Evidence Exhibit",
                    options=exhibits,
                    format_func=lambda p: f"[{'GENUINE' if 'real' in str(p) else 'SYNTHETIC'}] {p.name}",
                )
                if uploaded_file is None:
                    target_path = selected_exhibit
                    is_temp = False

    with col_src2:
        if target_path and target_path.is_file():
            meta = get_audio_metadata(target_path)
            sha256_hash = compute_file_sha256(target_path)
            evid_id = generate_evidence_id(target_path.name, sha256_hash)

            st.markdown("### Exhibit Metadata & Chain of Custody")
            st.markdown(f"**Evidence ID:** `{evid_id}`")
            st.markdown(f"**Filename:** `{target_path.name}`")
            st.markdown(f"**Format:** {meta.get('format', 'Unknown').upper()} | **Size:** {meta.get('file_size_bytes', 0):,} bytes")
            st.markdown(f"**Original Sample Rate:** {meta.get('sample_rate', 0):,} Hz | **Channels:** {meta.get('channels', 1)}")

            st.markdown("**Cryptographic SHA-256 Intake Digest:**")
            st.markdown(f'<div class="hash-box">{sha256_hash}</div>', unsafe_allow_html=True)

            st.audio(str(target_path))

    if not target_path:
        st.info("👆 Upload an audio file or select an exhibit from the catalog above to commence examination.")
        return

    st.markdown("---")

    # Examination Configuration
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        examiner = st.text_input("Lead Forensic Examiner", value="Forensic Analyst")
    with col_cfg2:
        case_ref = st.text_input("Case Reference Number", value=f"CASE-{datetime.date.today().strftime('%Y%m%d')}-001")
    with col_cfg3:
        primary_model = st.selectbox("Primary Classification Pipeline", options=["calibrated_svm", "random_forest", "logistic_regression"])

    run_btn = st.button("🚀 Run Complete Forensic Authenticity Examination", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing non-destructive 12-step forensic analysis pipeline..."):
            try:
                res = predict_audio(
                    target_path,
                    model_name=primary_model,
                    run_multi_model=True,
                    generate_visuals=True,
                )
                rep_md = create_forensic_report(res, case_number=case_ref, examiner_name=examiner, generate_pdf=True)
                st.session_state["last_analysis"] = res
                st.session_state["last_report"] = rep_md
                st.session_state["last_file_stem"] = target_path.stem
                st.success("Examination successfully completed and recorded to immutable audit trail!")
            except Exception as e:
                st.error(f"Examination encountered an error: {e}")
                return

    # Render Results if available in session
    if "last_analysis" in st.session_state:
        res = st.session_state["last_analysis"]
        rep_md = st.session_state.get("last_report", "")
        file_stem = st.session_state.get("last_file_stem", "evidence")

        st.markdown("---")
        st.markdown("## Forensic Examination Findings")

        pred_label = res["predicted_label"]
        status = res["status"]

        # Classification Callout
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.markdown("**Classification Finding:**")
            if pred_label == "GENUINE":
                st.markdown('<span class="badge-real">GENUINE SPEECH</span>', unsafe_allow_html=True)
            elif pred_label == "SYNTHETIC":
                st.markdown('<span class="badge-fake">SYNTHETIC / DEEPFAKE</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-indet">INDETERMINATE ({status})</span>', unsafe_allow_html=True)

        with col_res2:
            st.markdown("**Synthetic Posterior Probability:**")
            p_synth = res.get("calibrated_probability_synthetic")
            st.markdown(f"### {p_synth:.2%}" if p_synth is not None else "### N/A")

        with col_res3:
            st.markdown("**Genuine Posterior Probability:**")
            p_real = res.get("calibrated_probability_genuine")
            st.markdown(f"### {p_real:.2%}" if p_real is not None else "### N/A")

        with col_res4:
            st.markdown("**Classifier Confidence:**")
            st.markdown(f"### {res.get('confidence', 0.0):.2%}")

        # Assessment callout
        st.markdown(
            f'<div class="callout-card"><b>Forensic Assessment Statement:</b><br>{res["forensic_assessment"]}</div>',
            unsafe_allow_html=True,
        )

        # Quality & Domain screening tabs
        tab_spec, tab_models, tab_quality, tab_report = st.tabs(
            [
                "📈 Spectrogram & Visual Evidence",
                "⚖️ Multi-Model Consensus",
                "🔍 Signal Quality & OOD Screening",
                "📄 Formal Forensic Report & Export",
            ]
        )

        with tab_spec:
            visuals = res.get("visualizations", {})
            if "overview" in visuals and Path(visuals["overview"]).is_file():
                st.image(visuals["overview"], caption="Forensic Multi-Panel Overview (Waveform, STFT, Mel, MFCC, Dynamics)", use_container_width=True)
            else:
                st.warning("Visual figures were not generated or are unavailable.")

        with tab_models:
            st.markdown("### Multi-Model Comparison & Disagreement Analysis")
            multi = res.get("multi_model_analysis")
            if multi and "individual_models" in multi:
                m_data = []
                for m_k, m_v in multi["individual_models"].items():
                    if isinstance(m_v, dict) and "predicted_label" in m_v:
                        m_data.append({
                            "Classifier Pipeline": m_k.upper(),
                            "Predicted Class": m_v["predicted_label"],
                            "Synthetic Probability": f"{m_v.get('probability_synthetic', 0.0):.2%}" if m_v.get("probability_synthetic") is not None else "N/A",
                            "Genuine Probability": f"{m_v.get('probability_genuine', 0.0):.2%}" if m_v.get("probability_genuine") is not None else "N/A",
                            "Confidence": f"{m_v.get('confidence', 0.0):.2%}",
                            "Decision Score": f"{m_v.get('decision_score', 0.0):.4f}" if m_v.get("decision_score") is not None else "N/A",
                        })
                st.table(m_data)
                st.info(f"Consensus Assessment: {multi['consensus_assessment']}")

        with tab_quality:
            st.markdown("### Acoustic Signal Screening")
            q = res.get("quality_assessment", {})
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("Duration", f"{q.get('duration_seconds', 0.0):.2f}s")
            col_q2.metric("Clipping Ratio", f"{q.get('clipping_ratio', 0.0):.2%}")
            col_q3.metric("Silence Ratio", f"{q.get('silence_ratio', 0.0):.2%}")
            col_q4.metric("Estimated SNR", f"{q.get('estimated_snr_db', 0.0):.1f} dB")

            ood = res.get("distribution_assessment", {})
            st.markdown("### Feature Distribution Bounds (OOD Detection)")
            st.markdown(f"**Domain Classification:** `{ood.get('status', 'IN_DISTRIBUTION')}`")
            st.markdown(f"**Maximum Feature Z-Score:** `{ood.get('max_z_score', 0.0)}` (Threshold: {ood.get('max_z_threshold', 6.5)})")
            st.markdown(f"**Mean Feature Z-Score:** `{ood.get('mean_z_score', 0.0)}` (Threshold: {ood.get('mean_z_threshold', 2.2)})")

        with tab_report:
            st.markdown("### Formal Forensic Report")
            rep_path = root / "results" / "reports" / f"FORENSIC_EXAMINATION_REPORT_{file_stem}.md"
            pdf_path = root / "results" / "reports" / f"FORENSIC_EXAMINATION_REPORT_{file_stem}.pdf"

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Download Formal Report (Markdown .md)",
                    data=rep_md,
                    file_name=f"FORENSIC_REPORT_{file_stem}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_dl2:
                if pdf_path.is_file():
                    st.download_button(
                        "📄 Download Formal Court-Ready Report (PDF .pdf)",
                        data=pdf_path.read_bytes(),
                        file_name=f"FORENSIC_REPORT_{file_stem}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.info("PDF document compilation in progress or unavailable.")

            with st.expander("Preview Complete 10-Section Markdown Report", expanded=True):
                st.markdown(rep_md)


def render_catalog_page():
    st.markdown('<div class="main-header">Evidence Exhibit Catalog</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Overview of 66 preserved forensic audio exhibits and demonstration recordings.</div>',
        unsafe_allow_html=True,
    )

    manifest_path = root / "dataset" / "recording_level_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        table_rows = []
        for item in manifest:
            table_rows.append({
                "Filename": item.get("filename"),
                "True Ground Truth": item.get("label_name"),
                "Speaker": item.get("speaker", "").title(),
                "Duration (s)": round(item.get("duration_seconds", 0.0), 2),
                "Sample Rate": f"{item.get('sample_rate', 0):,} Hz",
                "SHA-256 Intake Hash": item.get("sha256", "")[:16] + "...",
            })
        st.dataframe(table_rows, use_container_width=True, height=450)
    else:
        st.warning("Recording-level manifest not found.")


def render_audit_page():
    st.markdown('<div class="main-header">Forensic Audit Trail & Chain of Custody</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Immutable, append-only log of all computational transformations and examination findings.</div>',
        unsafe_allow_html=True,
    )

    audit_path = root / "results" / "reports" / "forensic_audit_trail.jsonl"
    if audit_path.is_file():
        events = []
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass

        if events:
            st.markdown(f"**Total Registered Forensic Events:** `{len(events)}`")
            for ev in reversed(events[-15:]):
                with st.expander(f"📌 {ev.get('timestamp_utc', 'N/A')} — {ev.get('event_data', {}).get('evidence_id', 'EVID')} ({ev.get('event_data', {}).get('predicted_label', 'UNKNOWN')})"):
                    st.json(ev)
        else:
            st.info("Audit trail log is currently empty.")
    else:
        st.info("No examination events recorded yet. Run an analysis to generate audit events.")


def render_methodology_page():
    st.markdown('<div class="main-header">Forensic Methodology & Legal Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### Adherence to ISO/IEC 27037 Technical Principles
        This platform adheres strictly to the digital evidence principles articulated in **ISO/IEC 27037:2012**:
        - **Identification:** Preserves evidence provenance, subject IDs, and metadata.
        - **Collection & Acquisition:** Cryptographic SHA-256 digests verify original evidence immutability.
        - **Preservation:** Audio standardisation operates strictly on floating-point memory buffers or derivative directories (`dataset/preprocessed/`). Raw evidence is treated as read-only.
        
        ### Scientific & Legal Limitations
        1. **Probabilistic Nature:** Statistical model predictions represent analytical indicators of acoustic anomaly, not conclusive proof of fabrication or innocence.
        2. **Acoustic Boundaries:** Lossy codecs (e.g. telephony bandpass < 3,400 Hz) and severe background noise (< 15 dB SNR) degrade feature reliability.
        3. **Expert In-the-Loop:** In criminal investigative contexts, automated screening must be corroborated by forensic examiners, linguistic transcripts, and contextual evidence.
        """
    )


if __name__ == "__main__":
    main()
