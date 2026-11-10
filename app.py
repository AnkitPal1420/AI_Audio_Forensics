"""AI-Based Voice Analysis for Detecting Synthetic Audio in Criminal Investigations.

Phase 5 Forensic Web Dashboard:
- Integrated Case Management & Digital Evidence Repository (SQLite).
- Non-destructive 12-step Audio Examination Workspace.
- Multi-Model AI Detection (Ensemble, Calibrated SVM, Random Forest, Logistic Regression).
- Cryptographic SHA-256 Evidence Integrity Verification (ISO/IEC 27037).
- Publication-Grade Acoustic Spectrogram & Waveform Visualizations.
- Automated Forensic Report Center (Markdown & Court-Ready PDF).
- Historical Case Audit Trail and KPI Analytics.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import tempfile
import json
import sqlite3
import streamlit as st
import numpy as np

# Antigravity audio forensics modules
from database.database import ForensicDatabase
from database.models import (
    AnalysisFinding,
    AnalysisRun,
    EvidenceComparisonRecord,
    ForensicCase,
    ForensicEvidence,
    ForensicExamination,
    IntegrityRecord,
)
from src.anomaly_detection import detect_acoustic_anomalies
from src.audio_prediction import align_feature_vector, predict_audio
from src.audio_preprocessing import get_audio_metadata, load_audio
from src.audio_quality import assess_audio_quality, assess_feature_distribution
from src.batch_processing import process_audio_batch
from src.deepfake_detection import DeepfakeDetectionEngine
from src.evidence_correlation import correlate_exhibits
from src.explainable_ai import explain_prediction
from src.feature_extraction import CANONICAL_FEATURE_NAMES, extract_basic_features
from src.forensic_integrity import (
    compute_file_sha256,
    generate_evidence_id,
    get_environment_metadata,
    verify_evidence_immutability,
)
from src.forensic_report import create_forensic_report
from src.generator_analysis import profile_generator_artifacts
from src.speaker_comparison import compare_acoustic_profiles
from src.spectrogram_analysis import compute_spectral_trajectories, generate_forensic_visualizations
from src.storage_manager import get_storage_paths, ingest_evidence_file, seed_default_cases_and_exhibits
from src.tampering_analysis import analyze_audio_tampering, compute_frame_stft
from src.utils import project_root

# Page Configuration
st.set_page_config(
    page_title="Forensic Audio Intelligence | Phase 5 Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Forensic Laboratory Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }

    code, .font-mono, pre {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
    }

    /* Forensic Lab Header */
    .lab-header {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .lab-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #f0f6fc 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .lab-subtitle {
        font-size: 0.9rem;
        color: #8b949e;
        line-height: 1.4;
    }

    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f0f6fc;
        letter-spacing: -0.02em;
    }

    /* Forensic Status Pills */
    .pill-genuine {
        background: rgba(35, 134, 54, 0.2);
        border: 1px solid #2ea043;
        color: #3fb950;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-synthetic {
        background: rgba(218, 54, 51, 0.2);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-indeterminate {
        background: rgba(187, 128, 9, 0.2);
        border: 1px solid #bb8009;
        color: #d29922;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Cryptographic Hash Container */
    .hash-display {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #38bdf8;
        word-break: break-all;
    }

    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(180deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(180deg, #0369a1 0%, #075985 100%) !important;
        border-color: #7dd3fc !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

root = project_root()
storage = get_storage_paths()
db = ForensicDatabase()
seed_default_cases_and_exhibits(db)
engine = DeepfakeDetectionEngine()


def render_styled_table(data: list[dict]):
    """Renders tabular data as an Apple-styled HTML table without requiring pandas."""
    if not data:
        st.info("No records to display.")
        return
    headers = list(data[0].keys())
    th_html = "".join(
        f"<th style='padding: 8px 12px; text-align: left; font-size: 11px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid rgba(255,255,255,0.1);'>{h}</th>"
        for h in headers
    )
    rows_html = []
    for row in data:
        td_html = "".join(
            f"<td style='padding: 9px 12px; font-size: 13px; color: #f5f5f7; border-bottom: 1px solid rgba(255,255,255,0.05);'>{row.get(h, '')}</td>"
            for h in headers
        )
        rows_html.append(f"<tr>{td_html}</tr>")
    table_html = f"""
    <div style='overflow-x: auto; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; margin: 8px 0 16px 0;'>
      <table style='width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, sans-serif;'>
        <thead><tr style='background: rgba(255,255,255,0.03);'>{th_html}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #f0f6fc;">VOICE FORENSICS</div>
                    <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">PHASE 7 FORENSIC INTELLIGENCE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Forensic Workspace",
            [
                "🏠 Dashboard Overview",
                "📁 Case Management",
                "📤 Evidence Ingestion & Upload",
                "🔬 Audio Examination Workspace",
                "🤖 AI Detection & Consensus",
                "🔒 Evidence Integrity Verification",
                "📄 Forensic Report Center",
                "📜 Examination Audit Trail",
                "📊 Scientific Validation & Reliability",
                "🧠 Forensic Intelligence & Manipulation",
                "⚙️ System Settings & Diagnostics",
            ],
            index=0,
        )

        st.markdown("---")

        # Active Model Pill
        avail = engine.get_available_models()
        st.markdown(
            f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px; font-size: 0.78rem;">
                <div style="color: #8b949e; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Engine Status</div>
                <div style="color: #3fb950; font-weight: 700;">● Online ({len(avail)} models loaded)</div>
                <div style="color: #8b949e; margin-top: 4px;">Standard: ISO/IEC 27037 Technical Principles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route Page
    if page == "🏠 Dashboard Overview":
        render_home_dashboard()
    elif page == "📁 Case Management":
        render_case_management()
    elif page == "📤 Evidence Ingestion & Upload":
        render_evidence_upload()
    elif page == "🔬 Audio Examination Workspace":
        render_audio_examination()
    elif page == "🤖 AI Detection & Consensus":
        render_ai_detection()
    elif page == "🔒 Evidence Integrity Verification":
        render_integrity_verification()
    elif page == "📄 Forensic Report Center":
        render_report_center()
    elif page == "📜 Examination Audit Trail":
        render_examination_history()
    elif page == "📊 Scientific Validation & Reliability":
        render_scientific_validation_workspace()
    elif page == "🧠 Forensic Intelligence & Manipulation":
        render_phase7_forensic_intelligence()
    else:
        render_settings_and_diagnostics()


# =============================================================================
# 1. Dashboard Overview
# =============================================================================
def render_home_dashboard():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Forensic Laboratory Overview</div>
            <div class="lab-subtitle">
                Centralized investigative dashboard for digital voice analysis, deepfake detection, and cryptographic evidence tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = db.get_dashboard_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Cases</div><div class="metric-val">{metrics["total_cases"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Exhibits Logged</div><div class="metric-val">{metrics["total_evidence"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Examinations</div><div class="metric-val">{metrics["total_examinations"]}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Genuine Speech</div><div class="metric-val" style="color: #3fb950;">{metrics["genuine_count"]}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Synthetic / Deepfake</div><div class="metric-val" style="color: #f85149;">{metrics["synthetic_count"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_recent, col_models = st.columns([1.4, 1], gap="medium")

    with col_recent:
        st.markdown("### 📋 Recent Forensic Cases")
        cases = db.list_cases()
        if cases:
            case_data = []
            for c in cases[:6]:
                evids = db.list_evidence_by_case(c["case_id"])
                case_data.append({
                    "Case ID": c["case_id"],
                    "Case Title": c["title"],
                    "Investigator": c["investigator"],
                    "Exhibits": len(evids),
                    "Status": c["status"],
                    "Created": c["created_at"][:10],
                })
            render_styled_table(case_data)
        else:
            st.info("No cases created yet.")

    with col_models:
        st.markdown("### 🤖 Model Verification Status")
        models = [
            ("Ensemble (Soft Voting)", "models/ensemble_voting_model.joblib", "Verified Active", "#3fb950"),
            ("Calibrated SVM (RBF)", "models/calibrated_svm_model.joblib", "Platt Scaled (Verified)", "#3fb950"),
            ("Random Forest (100 Trees)", "models/rf_model.joblib", "Verified Active", "#3fb950"),
            ("Logistic Regression", "models/logistic_regression_model.joblib", "Linear Baseline", "#38bdf8"),
        ]
        for name, path_s, status, color in models:
            st.markdown(
                f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; font-size: 0.88rem; color: #f0f6fc;">{name}</div>
                        <div style="font-family: monospace; font-size: 0.75rem; color: #8b949e;">{path_s}</div>
                    </div>
                    <span style="font-size: 0.78rem; font-weight: 700; color: {color};">{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# 2. Case Management
# =============================================================================
def render_case_management():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Forensic Case Management</div>
            <div class="lab-subtitle">
                Create incident investigations, track evidence custody, and manage multi-exhibit criminal audio files.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_list, t_create = st.tabs(["🗂️ Active Case Registry", "➕ Create New Forensic Case"])

    with t_create:
        with st.form("new_case_form"):
            c1, c2 = st.columns(2)
            with c1:
                auto_id = f"CASE-{datetime.date.today().strftime('%Y%m%d')}-{len(db.list_cases()) + 1:03d}"
                case_id = st.text_input("Case Reference Identifier", value=auto_id)
                case_title = st.text_input("Case Title / Subject", placeholder="e.g. VIP Voice Cloning Extortion Incident")
            with c2:
                investigator = st.text_input("Lead Forensic Examiner", value="Special Agent Analyst")
                case_status = st.selectbox("Initial Case Status", ["ACTIVE", "PENDING_REVIEW", "ARCHIVED"])

            description = st.text_area("Case Incident Summary & Investigative Context", placeholder="Document the origin of evidence, acquisition warrants, and acoustic context...")

            submitted = st.form_submit_button("Register Forensic Case")
            if submitted:
                if not case_id or not case_title:
                    st.error("Case Reference ID and Title are required.")
                elif db.get_case(case_id):
                    st.error(f"Case with ID '{case_id}' already exists.")
                else:
                    new_case = ForensicCase(
                        case_id=case_id,
                        title=case_title,
                        description=description,
                        investigator=investigator,
                        status=case_status,
                    )
                    db.create_case(new_case)
                    db.log_audit_event("CASE_CREATED", new_case.to_dict(), case_id=case_id)
                    st.success(f"Forensic case '{case_id}' successfully created!")
                    st.rerun()

    with t_list:
        c_search, c_filter = st.columns([2, 1])
        with c_search:
            search_term = st.text_input("Search Cases", placeholder="Filter by Case ID, Title, or Investigator...", label_visibility="collapsed")
        with c_filter:
            status_filter = st.selectbox("Status Filter", ["ALL", "ACTIVE", "PENDING_REVIEW", "CLOSED", "ARCHIVED"], label_visibility="collapsed")

        cases = db.list_cases(search=search_term, status=status_filter)
        if cases:
            for c in cases:
                with st.expander(f"📁 {c['case_id']} — {c['title']} ({c['status']})", expanded=False):
                    col_info1, col_info2 = st.columns([1.5, 1])
                    with col_info1:
                        st.markdown(f"**Investigator:** {c['investigator']}")
                        st.markdown(f"**Description:** {c['description'] or 'No summary provided.'}")
                        st.markdown(f"**Date Created (UTC):** `{c['created_at']}`")
                    with col_info2:
                        new_stat = st.selectbox("Update Status", ["ACTIVE", "PENDING_REVIEW", "CLOSED", "ARCHIVED"], index=["ACTIVE", "PENDING_REVIEW", "CLOSED", "ARCHIVED"].index(c["status"]), key=f"stat_{c['case_id']}")
                        if new_stat != c["status"]:
                            db.update_case_status(c["case_id"], new_stat)
                            st.success(f"Updated status of {c['case_id']} to {new_stat}")
                            st.rerun()

                    # Exhibits under this case
                    evs = db.list_evidence_by_case(c["case_id"])
                    st.markdown(f"#### Digital Evidence Exhibits ({len(evs)})")
                    if evs:
                        ev_table = []
                        for e in evs:
                            ev_table.append({
                                "Evidence ID": e["evidence_id"],
                                "Filename": e["original_filename"],
                                "Duration (s)": e["duration_seconds"],
                                "Sampling Rate": f"{e['sample_rate']:,} Hz",
                                "SHA-256 Digest": e["sha256_hash"][:16] + "...",
                            })
                        render_styled_table(ev_table)
                    else:
                        st.info("No audio evidence exhibits uploaded to this case yet.")
        else:
            st.info("No forensic cases found matching the criteria.")


# =============================================================================
# 3. Audio Evidence Upload
# =============================================================================
def render_evidence_upload():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Audio Evidence Ingestion & Upload</div>
            <div class="lab-subtitle">
                Securely ingest digital audio exhibits, compute cryptographic intake digests, and catalog provenance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cases = db.list_cases()
    if not cases:
        st.warning("No forensic cases exist. Please create a case first under 'Case Management'.")
        return

    case_options = {f"{c['case_id']} — {c['title']}": c["case_id"] for c in cases}
    selected_case_label = st.selectbox("Assign to Forensic Case", options=list(case_options.keys()))
    assigned_case_id = case_options[selected_case_label]

    uploaded_files = st.file_uploader(
        "Submit Audio Evidence Exhibits",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        help="WAV, MP3, FLAC, and OGG containers supported.",
    )

    if uploaded_files and st.button("🔒 Securely Ingest Evidence Files"):
        progress_bar = st.progress(0, text="Ingesting evidence...")
        ingested_count = 0

        for i, up_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Ingesting {up_file.name}...")
            bytes_data = up_file.read()
            dest_path, evid_id, sha256_hash, meta = ingest_evidence_file(
                bytes_data, up_file.name, assigned_case_id
            )

            # Register in database
            ev_record = ForensicEvidence(
                evidence_id=evid_id,
                case_id=assigned_case_id,
                original_filename=up_file.name,
                stored_path=str(dest_path),
                file_size_bytes=meta.get("file_size_bytes", len(bytes_data)),
                duration_seconds=round(meta.get("duration_seconds", 0.0), 2),
                sample_rate=meta.get("sample_rate", 16000),
                channels=meta.get("channels", 1),
                container_format=meta.get("format", up_file.name.split(".")[-1].upper()),
                sha256_hash=sha256_hash,
            )
            db.add_evidence(ev_record)
            db.log_audit_event("EVIDENCE_INGESTED", ev_record.to_dict(), evidence_id=evid_id, case_id=assigned_case_id)
            ingested_count += 1

        st.success(f"Successfully ingested and cryptographically fingerprinted {ingested_count} evidence file(s) into case '{assigned_case_id}'.")


# =============================================================================
# 4. Audio Examination Workspace
# =============================================================================
def render_audio_examination():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Audio Examination Workspace</div>
            <div class="lab-subtitle">
                Interactive acoustic signal processing: waveforms, STFT linear spectrograms, 128-band Mel representations, and MFCC dynamics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_evidence = db.list_all_evidence()
    if not all_evidence:
        st.warning("No evidence exhibits found in database. Ingest audio under 'Evidence Ingestion' or select an exhibit.")
        return

    ev_options = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in all_evidence}
    selected_ev_label = st.selectbox("Select Exhibit for Examination", options=list(ev_options.keys()))
    exhibit = ev_options[selected_ev_label]

    stored_file = Path(exhibit["stored_path"])
    if not stored_file.is_file():
        st.error(f"Stored evidence file not found at: {stored_file}")
        return

    # Playback & Metadata Bar
    col_play, col_meta = st.columns([1.2, 1], gap="medium")
    with col_play:
        st.markdown("#### 🎧 Audio Playback")
        st.audio(str(stored_file))
    with col_meta:
        st.markdown("#### 📋 Exhibit Metadata")
        st.markdown(f"**Evidence ID:** `{exhibit['evidence_id']}`")
        st.markdown(f"**Format:** {exhibit['container_format']} | **Duration:** {exhibit['duration_seconds']:.2f}s | **Sampling:** {exhibit['sample_rate']:,} Hz")
        st.markdown(f'<div class="hash-display">{exhibit["sha256_hash"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Generate & Display Visualizations
    with st.spinner("Computing Short-Time Fourier Transforms and Mel-frequency filterbanks..."):
        wave, sr = load_audio(stored_file, sample_rate=16_000, mono=True, normalize=True)
        visuals = generate_forensic_visualizations(
            wave, sr, output_dir=storage["visualizations"], file_stem=Path(exhibit["original_filename"]).stem
        )
        trajectories = compute_spectral_trajectories(wave, sr)

    t_overview, t_trajectories, t_stats = st.tabs(["📊 Master 6-Panel Spectrogram", "📈 Trajectory Dynamics", "🔢 Acoustic Numerical Summary"])

    with t_overview:
        if "overview" in visuals and Path(visuals["overview"]).is_file():
            st.image(str(visuals["overview"]), caption="Forensic Overview (16 kHz Mono, Hann 2048, Hop 512, Slaney 128 Mel bands)", use_container_width=True)

    with t_trajectories:
        st.markdown("#### Frame-Level Acoustic Trajectories")
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.line_chart({"Centroid (Hz)": trajectories["centroid"][:400], "Rolloff (Hz)": trajectories["rolloff"][:400]})
        with c_tr2:
            st.line_chart({"RMS Energy": trajectories["rms"][:400], "ZCR": trajectories["zcr"][:400]})

    with t_stats:
        st.markdown("#### Acoustic & Spectral Feature Statistics (Distinguished from Trained Features)")
        st.caption("Note: Below are exploratory signal metrics. Baseline models specifically employ 26 standardized Mel & spectral features.")
        c_st1, c_st2, c_st3, c_st4 = st.columns(4)
        c_st1.metric("Mean Centroid", f"{np.mean(trajectories['centroid']):.1f} Hz")
        c_st2.metric("Mean Rolloff (85%)", f"{np.mean(trajectories['rolloff']):.1f} Hz")
        c_st3.metric("Mean RMS Energy", f"{np.mean(trajectories['rms']):.4f}")
        c_st4.metric("Mean ZCR", f"{np.mean(trajectories['zcr']):.4f}")


# =============================================================================
# 5. AI Detection & Consensus
# =============================================================================
def render_ai_detection():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">AI Detection & Multi-Model Consensus Engine</div>
            <div class="lab-subtitle">
                Calibrated machine learning inference across multiple classifiers with automated disagreement analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_evidence = db.list_all_evidence()
    if not all_evidence:
        st.warning("No evidence exhibits available.")
        return

    ev_options = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in all_evidence}
    selected_label = st.selectbox("Choose Exhibit for Detection", options=list(ev_options.keys()))
    exhibit = ev_options[selected_label]
    stored_file = Path(exhibit["stored_path"])

    c_cfg1, c_cfg2 = st.columns([1.5, 1])
    with c_cfg1:
        model_name = st.selectbox(
            "Primary Classification Pipeline",
            options=["ensemble", "calibrated_svm", "random_forest", "logistic_regression"],
            help="Ensemble: Soft voting across SVM, RF, and LR. Calibrated SVM: Platt-scaled RBF SVM.",
        )
    with c_cfg2:
        examiner = st.text_input("Examiner Name", value="Lead Forensic Examiner")

    if st.button("🚀 Execute Calibrated Forensic Detection"):
        with st.spinner("Executing non-destructive 12-step detection pipeline..."):
            try:
                res = predict_audio(
                    stored_file,
                    model_name=model_name,
                    run_multi_model=True,
                    generate_visuals=True,
                    output_dir=storage["visualizations"],
                )

                # Generate reports
                rep_md_path = storage["reports"] / f"FORENSIC_REPORT_{Path(exhibit['original_filename']).stem}.md"
                create_forensic_report(
                    res,
                    case_number=exhibit["case_id"],
                    examiner_name=examiner,
                    output_path=rep_md_path,
                    generate_pdf=True,
                )
                rep_pdf_path = rep_md_path.with_suffix(".pdf")

                # Record into SQLite database
                exam_id = f"EXAM-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                q = res.get("quality_assessment", {})
                ood = res.get("distribution_assessment", {})
                multi = res.get("multi_model_analysis", {})

                exam_record = ForensicExamination(
                    examination_id=exam_id,
                    evidence_id=exhibit["evidence_id"],
                    case_id=exhibit["case_id"],
                    model_name=model_name,
                    predicted_label=res["predicted_label"],
                    confidence=res.get("confidence", 0.0),
                    prob_synthetic=res.get("calibrated_probability_synthetic"),
                    prob_genuine=res.get("calibrated_probability_genuine"),
                    decision_score=res.get("decision_score"),
                    status=res.get("status", "COMPLETED"),
                    estimated_snr_db=q.get("estimated_snr_db"),
                    clipping_ratio=q.get("clipping_ratio"),
                    silence_ratio=q.get("silence_ratio"),
                    is_in_distribution=ood.get("is_in_distribution", True),
                    max_z_score=ood.get("max_z_score"),
                    multi_model_consensus=multi.get("consensus_label") if multi else None,
                    forensic_assessment=res.get("forensic_assessment", ""),
                    overview_figure_path=res.get("visualizations", {}).get("overview"),
                    report_md_path=str(rep_md_path),
                    report_pdf_path=str(rep_pdf_path) if rep_pdf_path.is_file() else None,
                )
                db.add_examination(exam_record)
                db.log_audit_event("EXAMINATION_COMPLETED", exam_record.to_dict(), evidence_id=exhibit["evidence_id"], case_id=exhibit["case_id"])

                st.session_state["active_exam"] = res
                st.session_state["active_exam_db"] = exam_record
                st.success("Forensic detection completed and recorded in database!")
            except Exception as e:
                st.error(f"Inference failed: {e}")

    # Render Current Finding
    if "active_exam" in st.session_state:
        res = st.session_state["active_exam"]
        pred_label = res["predicted_label"]

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        pill_class = "pill-genuine" if pred_label == "GENUINE" else ("pill-synthetic" if pred_label == "SYNTHETIC" else "pill-indeterminate")

        st.markdown(
            f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 0.75rem; color: #8b949e; font-weight: 700; text-transform: uppercase;">Primary Finding</div>
                        <span class="{pill_class}">● {pred_label}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #8b949e; font-weight: 700; text-transform: uppercase;">Confidence</div>
                        <div style="font-size: 1.75rem; font-weight: 800; color: #f0f6fc;">{res.get('confidence', 0.0):.2%}</div>
                    </div>
                </div>
                <div style="font-size: 0.92rem; color: #f0f6fc; line-height: 1.5;">
                    {res.get('forensic_assessment')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Multi-model table
        multi = res.get("multi_model_analysis")
        if multi and "individual_models" in multi:
            st.markdown("#### Multi-Model Comparison & Disagreement Assessment")
            m_data = []
            for m_k, m_v in multi["individual_models"].items():
                if isinstance(m_v, dict) and "predicted_label" in m_v:
                    m_data.append({
                        "Classifier": m_k.upper(),
                        "Predicted Class": m_v["predicted_label"],
                        "Synthetic Prob": f"{m_v.get('probability_synthetic', 0.0):.2%}" if m_v.get("probability_synthetic") is not None else "N/A",
                        "Genuine Prob": f"{m_v.get('probability_genuine', 0.0):.2%}" if m_v.get("probability_genuine") is not None else "N/A",
                        "Confidence": f"{m_v.get('confidence', 0.0):.2%}",
                    })
            render_styled_table(m_data)


# =============================================================================
# 6. Evidence Integrity Verification
# =============================================================================
def render_integrity_verification():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Evidence Integrity & Cryptographic Verification</div>
            <div class="lab-subtitle">
                On-demand SHA-256 bitstream re-hashing and tampering detection conforming to ISO/IEC 27037 standards.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_evidence = db.list_all_evidence()
    if not all_evidence:
        st.warning("No evidence exhibits available.")
        return

    ev_options = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in all_evidence}
    selected_label = st.selectbox("Select Exhibit for Immutability Audit", options=list(ev_options.keys()))
    exhibit = ev_options[selected_label]
    stored_file = Path(exhibit["stored_path"])

    recorded_hash = exhibit["sha256_hash"]

    c_rec1, c_rec2 = st.columns(2)
    with c_rec1:
        st.markdown("**Recorded Intake SHA-256:**")
        st.markdown(f'<div class="hash-display">{recorded_hash}</div>', unsafe_allow_html=True)
    with c_rec2:
        st.markdown("**Evidence Tracking ID:**")
        st.markdown(f"**`{exhibit['evidence_id']}`** (Stored at `{stored_file.name}`)")

    verifier_name = st.text_input("Verifying Agent", value="Automated Integrity Monitor")
    notes = st.text_input("Verification Notes / Reason", value="Routine Case Integrity Audit")

    if st.button("🔒 Recalculate SHA-256 and Verify Bitstream Immutability"):
        if not stored_file.is_file():
            st.error("Evidence file missing from disk! Integrity violation.")
            return

        recalc_hash = compute_file_sha256(stored_file)
        is_match = (recalc_hash.lower() == recorded_hash.lower())
        status_str = "VERIFIED" if is_match else "HASH_MISMATCH"

        record = IntegrityRecord(
            evidence_id=exhibit["evidence_id"],
            recorded_sha256=recorded_hash,
            checked_sha256=recalc_hash,
            status=status_str,
            verified_by=verifier_name,
            notes=notes,
        )
        db.record_integrity_check(record)
        db.log_audit_event("INTEGRITY_CHECK", record.to_dict(), evidence_id=exhibit["evidence_id"], case_id=exhibit["case_id"])

        if is_match:
            st.success("✅ CRYPTOGRAPHIC INTEGRITY VERIFIED: Current file bitstream strictly matches intake digest.")
        else:
            st.error("🚨 CRITICAL ALERT: SHA-256 DIGEST MISMATCH! The file on disk has been altered or corrupted.")

    st.markdown("---")
    st.markdown("#### 📜 Historical Integrity Audits for this Exhibit")
    checks = db.list_integrity_checks(evidence_id=exhibit["evidence_id"])
    if checks:
        render_styled_table(checks)
    else:
        st.info("No prior integrity audits recorded for this exhibit.")

    st.caption("Forensic Disclaimer: A verified matching hash establishes that the examined digital file has not been altered since intake. It does not independently establish the authenticity or origin of the acoustic content itself.")


# =============================================================================
# 7. Forensic Report Center
# =============================================================================
def render_report_center():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Forensic Report Center</div>
            <div class="lab-subtitle">
                Download formal 10-section examination reports in GitHub-flavored Markdown and court-ready PDF formats.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exams = db.list_examinations()
    if not exams:
        st.info("No completed examinations found. Run an analysis under 'AI Detection' to generate reports.")
        return

    ex_options = {f"[{ex['created_at'][:19]}] {ex['case_id']} — {ex['evidence_id']} ({ex['predicted_label']})": ex for ex in exams}
    selected_label = st.selectbox("Select Examination Report", options=list(ex_options.keys()))
    exam = ex_options[selected_label]

    md_path = Path(exam["report_md_path"]) if exam.get("report_md_path") else None
    pdf_path = Path(exam["report_pdf_path"]) if exam.get("report_pdf_path") else None

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        if md_path and md_path.is_file():
            st.download_button(
                "📥 Download Formal Report (Markdown .md)",
                data=md_path.read_text(encoding="utf-8"),
                file_name=md_path.name,
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.warning("Markdown report not found.")

    with c_dl2:
        if pdf_path and pdf_path.is_file():
            st.download_button(
                "📄 Download Court-Ready Report (PDF .pdf)",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("PDF document compilation unavailable.")

    if md_path and md_path.is_file():
        with st.expander("Preview Formal 10-Section Report", expanded=True):
            st.markdown(md_path.read_text(encoding="utf-8"))


# =============================================================================
# 8. Examination Audit Trail
# =============================================================================
def render_examination_history():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Examination Audit Trail & Event Logs</div>
            <div class="lab-subtitle">
                Comprehensive, append-only historical database of all computational transformations and forensic operations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_exams, t_events = st.tabs(["📊 Completed Examinations", "🔒 System Audit Events"])

    with t_exams:
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            lbl_filter = st.selectbox("Filter by Finding", ["ALL", "GENUINE", "SYNTHETIC", "INDETERMINATE"])
        with c_f2:
            case_filter = st.text_input("Filter by Case ID", placeholder="e.g. CASE-20261110-001")

        exams = db.list_examinations(label=lbl_filter, case_id=case_filter if case_filter else None)
        if exams:
            render_styled_table(exams)
        else:
            st.info("No examinations matching filter.")

    with t_events:
        events = db.list_audit_events(limit=50)
        if events:
            for ev in events:
                with st.expander(f"📌 [{ev['timestamp_utc']}] {ev['event_type']} (Case: {ev.get('case_id') or 'N/A'})"):
                    st.json(json.loads(ev["details_json"]))
        else:
            st.info("No audit events recorded.")


# =============================================================================
# 9. Scientific Validation & Model Reliability (Phase 6)
# =============================================================================
def render_scientific_validation_workspace():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">Scientific Validation & Model Reliability</div>
            <div class="lab-subtitle">
                Independent forensic audit, cross-dataset generalization benchmarks, probability calibration, and channel robustness analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status Overview Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
                <div style="font-size: 0.75rem; color: #86868b; text-transform: uppercase; font-weight: 600;">Feature Pipeline Parity</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #3fb950; margin: 4px 0;">VALIDATED (Δ = 0.0)</div>
                <div style="font-size: 0.8rem; color: #a1a1a6;">26 canonical features mathematically aligned.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <div style="font-size: 0.75rem; color: #86868b; text-transform: uppercase; font-weight: 600;">Speaker Independence</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin: 4px 0;">LOSO AUC: 77.58%</div>
                <div style="font-size: 0.8rem; color: #a1a1a6;">8-fold cross-validation on held-out subjects.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <div style="font-size: 0.75rem; color: #86868b; text-transform: uppercase; font-weight: 600;">Telephony Vulnerability</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #f85149; margin: 4px 0;">COLLAPSE (Recall: 0%)</div>
                <div style="font-size: 0.8rem; color: #a1a1a6;">G.711 bandpass attenuates vocoder artifacts.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    t_loso, t_cross, t_calib, t_robust, t_ood, t_export = st.tabs([
        "🧑‍🔬 Speaker Independence",
        "🌐 Cross-Dataset (FoR)",
        "⚖️ Calibration & Thresholds",
        "🔊 Robustness Stress Tests",
        "🛡️ OOD Manifold",
        "📥 Phase 6 Reports",
    ])

    with t_loso:
        st.markdown("#### 🧑‍🔬 8-Fold Leave-One-Speaker-Out (LOSO) Cross-Validation")
        st.caption("Each subject was held out entirely as test data while models were trained on the remaining 7 subjects.")
        loso_data = [
            {"Held-Out Subject": "biden", "Samples": 1055, "Accuracy": "57.35%", "Balanced Acc": "61.52%", "ROC-AUC": "0.7414", "Status": "Passable"},
            {"Held-Out Subject": "linus", "Samples": 1145, "Accuracy": "45.24%", "Balanced Acc": "47.53%", "ROC-AUC": "0.4798", "Status": "Severe Shift"},
            {"Held-Out Subject": "margot", "Samples": 534, "Accuracy": "81.65%", "Balanced Acc": "65.69%", "ROC-AUC": "0.7996", "Status": "Validated"},
            {"Held-Out Subject": "musk", "Samples": 1055, "Accuracy": "43.70%", "Balanced Acc": "49.15%", "ROC-AUC": "0.6626", "Status": "Acoustic Shift"},
            {"Held-Out Subject": "obama", "Samples": 1055, "Accuracy": "64.17%", "Balanced Acc": "68.15%", "ROC-AUC": "0.8684", "Status": "Validated"},
            {"Held-Out Subject": "ryan", "Samples": 548, "Accuracy": "96.72%", "Balanced Acc": "94.60%", "ROC-AUC": "0.9912", "Status": "Exceptional"},
            {"Held-Out Subject": "taylor", "Samples": 1055, "Accuracy": "54.60%", "Balanced Acc": "59.45%", "ROC-AUC": "0.7682", "Status": "Passable"},
            {"Held-Out Subject": "trump", "Samples": 1055, "Accuracy": "66.16%", "Balanced Acc": "69.75%", "ROC-AUC": "0.8952", "Status": "Validated"},
            {"Held-Out Subject": "8-FOLD MACRO MEAN", "Samples": 7502, "Accuracy": "63.70%", "Balanced Acc": "64.48%", "ROC-AUC": "0.7758", "Status": "Summary Benchmark"},
        ]
        render_styled_table(loso_data)

    with t_cross:
        st.markdown("#### 🌐 Generalization to External Benchmark (Fake-or-Real Benchmark)")
        st.caption("Evaluated on 300 independent, unseen audio files: 200 clean TTS and 100 acoustic re-recordings.")
        cross_data = [
            {"Model": "Calibrated SVM", "Dataset Subset": "FoR 2-sec Clean (TTS)", "Files": 200, "Accuracy": "51.50%", "Precision": "51.90%", "Recall": "41.00%", "Specificity": "62.00%", "F1": "0.4581"},
            {"Model": "Calibrated SVM", "Dataset Subset": "FoR Re-recorded Acoustic", "Files": 100, "Accuracy": "38.00%", "Precision": "20.00%", "Recall": "8.00%", "Specificity": "68.00%", "F1": "0.1143"},
            {"Model": "Random Forest", "Dataset Subset": "FoR 2-sec Clean (TTS)", "Files": 200, "Accuracy": "39.50%", "Precision": "37.04%", "Recall": "30.00%", "Specificity": "49.00%", "F1": "0.3315"},
            {"Model": "Random Forest", "Dataset Subset": "FoR Re-recorded Acoustic", "Files": 100, "Accuracy": "35.00%", "Precision": "24.14%", "Recall": "14.00%", "Specificity": "56.00%", "F1": "0.1772"},
            {"Model": "Logistic Regression", "Dataset Subset": "FoR 2-sec Clean (TTS)", "Files": 200, "Accuracy": "23.00%", "Precision": "1.79%", "Recall": "1.00%", "Specificity": "45.00%", "F1": "0.0128"},
            {"Model": "Logistic Regression", "Dataset Subset": "FoR Re-recorded Acoustic", "Files": 100, "Accuracy": "37.00%", "Precision": "0.00%", "Recall": "0.00%", "Specificity": "74.00%", "F1": "0.0000"},
        ]
        render_styled_table(cross_data)
        st.warning("Forensic Finding: Physical acoustic re-recording masks high-frequency vocoder cues, reducing SVM detection recall from 41% to 8%. Automated classification should be restricted to direct digital extractions.")

    with t_calib:
        st.markdown("#### ⚖️ Probability Calibration & Threshold Optimization")
        c_k1, c_k2, c_k3 = st.columns(3)
        c_k1.metric("Validation Brier Score", "0.0968", help="Lower is better, ideal=0.0")
        c_k2.metric("Validation ECE", "2.25%", help="Expected Calibration Error across 10 bins")
        c_k3.metric("Calibration Slope", "1.01", help="Ideal slope is 1.00")

        st.markdown("##### Operational Threshold Comparison (Test Partition: 2,110 slices)")
        thresh_data = [
            {"Threshold Level": "Default (0.50)", "Accuracy": "59.81%", "Balanced Acc": "63.96%", "Precision": "51.88%", "Recall": "94.18%", "Specificity": "33.75%", "False Positives": 795, "Evaluation": "Excessive False Accusation"},
            {"Threshold Level": "Balanced (0.65)", "Accuracy": "66.59%", "Balanced Acc": "69.02%", "Precision": "57.47%", "Recall": "86.70%", "Specificity": "51.33%", "False Positives": 584, "Evaluation": "Moderate Triage"},
            {"Threshold Level": "Forensic Recommended (0.80)", "Accuracy": "70.62%", "Balanced Acc": "70.61%", "Precision": "64.59%", "Recall": "70.55%", "Specificity": "70.67%", "False Positives": 352, "Evaluation": "55.7% Fewer False Alarms"},
            {"Threshold Level": "Strict Screening (0.85)", "Accuracy": "70.90%", "Balanced Acc": "69.82%", "Precision": "67.79%", "Recall": "61.98%", "Specificity": "77.67%", "False Positives": 268, "Evaluation": "High Courtroom Conservatism"},
        ]
        render_styled_table(thresh_data)

    with t_robust:
        st.markdown("#### 🔊 Channel Degradation Stress-Testing (9 Conditions)")
        robust_data = [
            {"Acoustic Channel Condition": "Clean Baseline (16 kHz)", "Accuracy": "60.00%", "Recall": "80.00%", "F1-Score": "0.6667", "Forensic Vulnerability": "None (Reference)"},
            {"Acoustic Channel Condition": "Gaussian Noise (SNR 30 dB)", "Accuracy": "65.00%", "Recall": "80.00%", "F1-Score": "0.6957", "Forensic Vulnerability": "Resilient"},
            {"Acoustic Channel Condition": "Gaussian Noise (SNR 20 dB)", "Accuracy": "50.00%", "Recall": "30.00%", "F1-Score": "0.3750", "Forensic Vulnerability": "Moderate Noise Masking"},
            {"Acoustic Channel Condition": "Gaussian Noise (SNR 10 dB)", "Accuracy": "45.00%", "Recall": "10.00%", "F1-Score": "0.1538", "Forensic Vulnerability": "Severe Noise Masking"},
            {"Acoustic Channel Condition": "G.711 Telephony (300-3400 Hz)", "Accuracy": "50.00%", "Recall": "0.00%", "F1-Score": "0.0000", "Forensic Vulnerability": "TOTAL BLIND SPOT"},
            {"Acoustic Channel Condition": "Lossy Codec (Mu-law 8-bit)", "Accuracy": "65.00%", "Recall": "90.00%", "F1-Score": "0.7200", "Forensic Vulnerability": "Resilient Companding"},
            {"Acoustic Channel Condition": "Reverberation (RT60 = 0.3s)", "Accuracy": "60.00%", "Recall": "60.00%", "F1-Score": "0.6000", "Forensic Vulnerability": "Partial Comb Smearing"},
            {"Acoustic Channel Condition": "Digital Clipping (Threshold 0.7)", "Accuracy": "65.00%", "Recall": "70.00%", "F1-Score": "0.6667", "Forensic Vulnerability": "Mild Harmonic Distortion"},
            {"Acoustic Channel Condition": "Narrowband Resampling (8 kHz)", "Accuracy": "60.00%", "Recall": "50.00%", "F1-Score": "0.5556", "Forensic Vulnerability": "Aliasing Degradation"},
        ]
        render_styled_table(robust_data)

        chart_img = root / "results" / "graphs" / "robustness_performance_comparison.png"
        if chart_img.is_file():
            st.image(str(chart_img), caption="Diagnostic Performance Decay under Environmental & Channel Stress")

    with t_ood:
        st.markdown("#### 🛡️ Out-of-Distribution Screening & Manifold Bounds")
        st.markdown(
            """
            Statistical screening uses regularized Ledoit-Wolf Mahalanobis distance ($D_M$) to reject exhibits that fall outside human vocal tract covariances.
            """
        )
        ood_table = [
            {"Audio Domain": "Training In-Domain (n=4,844)", "Mean Distance (D_M)": "3.91", "99th Percentile Threshold": "6.93", "Outlier Ratio": "1.00%"},
            {"Audio Domain": "Validation Slices (n=548)", "Mean Distance (D_M)": "3.71", "99th Percentile Threshold": "6.93", "Outlier Ratio": "0.36%"},
            {"Audio Domain": "Unseen Test Subjects (n=2,110)", "Mean Distance (D_M)": "4.43", "99th Percentile Threshold": "6.93", "Outlier Ratio": "1.94%"},
            {"Audio Domain": "External FoR Benchmark (n=300)", "Mean Distance (D_M)": "4.96", "99th Percentile Threshold": "6.93", "Outlier Ratio": "0.00%"},
        ]
        render_styled_table(ood_table)

    with t_export:
        st.markdown("#### 📥 Export Phase 6 Independent Validation Reports")
        reports = [
            ("PHASE_6_PRE_IMPLEMENTATION_AUDIT.md", "Phase 6.0 Pre-Implementation Audit"),
            ("PHASE_6_FEATURE_PIPELINE_VALIDATION_REPORT.md", "Phase 6.1 Feature Parity & Determinism"),
            ("PHASE_6_SPEAKER_INDEPENDENT_EVALUATION_REPORT.md", "Phase 6.2 Speaker Independence & LOSO"),
            ("PHASE_6_CROSS_DATASET_VALIDATION_REPORT.md", "Phase 6.3 Cross-Dataset (FoR Benchmark)"),
            ("PHASE_6_CALIBRATION_AND_THRESHOLD_REPORT.md", "Phase 6.4 Calibration & Threshold Tuning"),
            ("PHASE_6_AUDIO_ROBUSTNESS_REPORT.md", "Phase 6.5 Channel Robustness Stress Tests"),
            ("PHASE_6_OOD_AND_UNCERTAINTY_REPORT.md", "Phase 6.6 Out-of-Distribution Manifolds"),
            ("PHASE_6_FORENSIC_RELIABILITY_REPORT.md", "Phase 6.7 Error Analysis & Bootstrap CIs"),
        ]

        c_col1, c_col2 = st.columns(2)
        for idx, (fname, desc) in enumerate(reports):
            fpath = root / fname
            target_col = c_col1 if idx % 2 == 0 else c_col2
            with target_col:
                if fpath.is_file():
                    target_col.download_button(
                        f"📄 {desc}",
                        data=fpath.read_text(encoding="utf-8"),
                        file_name=fname,
                        mime="text/markdown",
                        key=f"dl_p6_{idx}",
                    )
                else:
                    target_col.info(f"{fname} (Pending generation)")


# =============================================================================
# 10. Phase 7: Forensic Intelligence & Audio Manipulation Analysis
# =============================================================================
def render_phase7_forensic_intelligence():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">🧠 Forensic AI Intelligence & Audio Manipulation Analysis</div>
            <div class="lab-subtitle">
                Advanced AI explainability (XAI), signal-level edit detection, unsupervised anomaly scoring, vocoder profiling, and multi-evidence case correlation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_evidence = db.list_all_evidence()
    if not all_evidence:
        st.warning("No evidence exhibits found in database. Ingest audio under 'Evidence Ingestion' to enable analysis.")
        return

    ev_options = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in all_evidence}
    selected_ev_label = st.selectbox("Select Primary Exhibit for Forensic Intelligence", options=list(ev_options.keys()), key="p7_primary_ev")
    exhibit = ev_options[selected_ev_label]
    stored_file = Path(exhibit["stored_path"])

    if not stored_file.is_file():
        st.error(f"Evidence file not found on disk at: {stored_file}")
        return

    t_xai, t_tamp, t_spk, t_gen, t_anom, t_sig, t_corr, t_p7_docs = st.tabs([
        "🔍 Explainable AI (XAI)",
        "⚡ Manipulation & Splicing",
        "👥 Speaker Comparison Baseline",
        "🤖 Generator Characterization",
        "🚨 Acoustic Anomaly Detection",
        "🔬 Advanced Signal Workspace",
        "🔗 Multi-Evidence Correlation",
        "📥 Phase 7 Reports",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Explainable AI (XAI)
    # -------------------------------------------------------------------------
    with t_xai:
        st.markdown("#### 🔍 Feature Attribution & Decision Explainability")
        st.error(
            "⚠️ **MODEL UNVALIDATED WARNING**: The underlying classification models have unvalidated generalization "
            "on unseen acoustic domains and G.711 telephony channels (FoR ROC-AUC 0.283 - 0.534). "
            "Feature attributions indicate which statistical acoustic parameters influenced the algorithmic output; "
            "they do NOT prove that the audio was synthetically synthesized or manipulated."
        )

        c_m1, c_m2 = st.columns([2, 1])
        with c_m1:
            chosen_model = st.selectbox(
                "Select Model to Explain",
                ["calibrated_svm", "random_forest", "logistic_regression"],
                index=0,
                key="p7_xai_model",
            )
        with c_m2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_xai_btn = st.button("🚀 Run Explainability Analysis", key="btn_run_xai")

        if run_xai_btn:
            with st.spinner("Computing Shapley sampling attributions and feature occlusion..."):
                waveform, sr = load_audio(stored_file, target_sr=16000, mono=True)
                raw_feats = extract_basic_features(waveform, sample_rate=sr, n_mfcc=20)
                feat_vec = align_feature_vector(raw_feats, engine.feature_names)

                xai_res = explain_prediction(
                    feat_vec,
                    evidence_id=exhibit["evidence_id"],
                    model_key=chosen_model,
                    compute_shapley=True,
                )

                # Persist Run
                run_rec = AnalysisRun(
                    run_id=xai_res.run_id,
                    case_id=exhibit["case_id"],
                    evidence_id=exhibit["evidence_id"],
                    analysis_type="XAI_EXPLANATION",
                    algorithm_name="explain_prediction",
                    algorithm_version="1.0.0",
                    software_versions_json=json.dumps({"numpy": np.__version__}),
                    parameters_json=json.dumps({"model_key": chosen_model, "method": xai_res.method_used}),
                    input_sha256=exhibit["sha256_hash"],
                    status="COMPLETED",
                    warnings_json=json.dumps([xai_res.unvalidated_model_warning]),
                    results_json=json.dumps({
                        "predicted_label": xai_res.predicted_label,
                        "prob_synthetic": xai_res.predicted_prob_synthetic,
                        "grouped_importance": xai_res.grouped_importance,
                    }),
                    output_files_json=json.dumps([]),
                )
                db.create_analysis_run(run_rec)
                db.create_analysis_findings_batch(xai_res.findings)

                # Display Results
                st.markdown("---")
                c_pred1, c_pred2, c_pred3 = st.columns(3)
                label_color = "#f85149" if xai_res.predicted_label == "SYNTHETIC" else "#3fb950"
                c_pred1.markdown(f"**Predicted Label**: <span style='color:{label_color}; font-weight:800;'>{xai_res.predicted_label}</span>", unsafe_allow_html=True)
                c_pred2.markdown(f"**Synthetic Probability**: `{xai_res.predicted_prob_synthetic:.4f}`")
                c_pred3.markdown(f"**Attribution Method**: `{xai_res.method_used}`")

                st.markdown("##### 📊 Top Feature Drivers toward SYNTHETIC Classification")
                if xai_res.top_synthetic_contributors:
                    synth_rows = [
                        {
                            "Feature Name": c["feature_name"],
                            "Value": f"{c['feature_value']:.4f}",
                            "Group": c["group"],
                            "Shapley Attribution": f"+{c['importance_score']:.4f}",
                            "Influence Direction": "Pushed Toward SYNTHETIC",
                        }
                        for c in xai_res.top_synthetic_contributors
                    ]
                    render_styled_table(synth_rows)
                else:
                    st.info("No features exerted positive attribution toward synthetic prediction.")

                st.markdown("##### 🛡️ Top Feature Drivers toward GENUINE Classification")
                if xai_res.top_genuine_contributors:
                    real_rows = [
                        {
                            "Feature Name": c["feature_name"],
                            "Value": f"{c['feature_value']:.4f}",
                            "Group": c["group"],
                            "Shapley Attribution": f"{c['importance_score']:.4f}",
                            "Influence Direction": "Pushed Toward GENUINE",
                        }
                        for c in xai_res.top_genuine_contributors
                    ]
                    render_styled_table(real_rows)
                else:
                    st.info("No features exerted negative attribution toward genuine prediction.")

                st.markdown("##### 📈 Grouped Feature Attribution Share")
                grp_rows = [{"Acoustic Dimension": k, "Relative Importance Share": f"{v * 100:.2f}%"} for k, v in xai_res.grouped_importance.items()]
                render_styled_table(grp_rows)

    # -------------------------------------------------------------------------
    # TAB 2: Audio Manipulation & Splicing
    # -------------------------------------------------------------------------
    with t_tamp:
        st.markdown("#### ⚡ Signal-Level Edit, Splice & Discontinuity Detection")
        st.info(
            "Detects micro-acoustic anomalies across 10ms-25ms analysis frames: "
            "sharp spectral flux surges, RMS step transitions, background noise floor shifts, "
            "and repeated waveform segments (audio clone-and-paste splicing)."
        )

        c_th1, c_th2, c_th3 = st.columns(3)
        flux_sigma = c_th1.slider("Spectral Flux Sigma Threshold (σ)", min_value=2.0, max_value=6.0, value=3.5, step=0.1)
        rms_sigma = c_th2.slider("RMS Step Sigma Threshold (σ)", min_value=2.0, max_value=6.0, value=3.5, step=0.1)
        noise_shift = c_th3.slider("Noise Floor Shift Threshold (dB)", min_value=3.0, max_value=12.0, value=6.0, step=0.5)

        if st.button("⚡ Execute Signal Tampering Analysis", key="btn_run_tamp"):
            with st.spinner("Extracting short-time spectral dynamics and tracking discontinuities..."):
                waveform, sr = load_audio(stored_file, target_sr=16000, mono=True)
                tamp_res = analyze_audio_tampering(
                    waveform,
                    sr,
                    evidence_id=exhibit["evidence_id"],
                    flux_sigma_threshold=flux_sigma,
                    rms_step_sigma_threshold=rms_sigma,
                    noise_floor_shift_db=noise_shift,
                )

                # Persist Run
                run_rec = AnalysisRun(
                    run_id=tamp_res.run_id,
                    case_id=exhibit["case_id"],
                    evidence_id=exhibit["evidence_id"],
                    analysis_type="AUDIO_MANIPULATION",
                    algorithm_name="analyze_audio_tampering",
                    algorithm_version="1.0.0",
                    software_versions_json=json.dumps({"numpy": np.__version__}),
                    parameters_json=json.dumps({"flux_sigma": flux_sigma, "rms_sigma": rms_sigma, "noise_shift_db": noise_shift}),
                    input_sha256=exhibit["sha256_hash"],
                    status="COMPLETED",
                    warnings_json=json.dumps([]),
                    results_json=json.dumps({"total_anomalies": tamp_res.total_anomalies}),
                    output_files_json=json.dumps([]),
                )
                db.create_analysis_run(run_rec)
                db.create_analysis_findings_batch(tamp_res.findings)

                # Summary Metrics
                c_k1, c_k2, c_k3, c_k4 = st.columns(4)
                c_k1.metric("Total Flagged Indicators", tamp_res.total_anomalies)
                flux_count = sum(1 for f in tamp_res.findings if f.indicator == "SPECTRAL_FLUX_DISCONTINUITY")
                rms_count = sum(1 for f in tamp_res.findings if f.indicator == "RMS_ENERGY_STEP")
                rep_count = sum(1 for f in tamp_res.findings if f.indicator == "REPEATED_SEGMENT")
                c_k2.metric("Spectral Flux Surges", flux_count)
                c_k3.metric("RMS Energy Steps", rms_count)
                c_k4.metric("Repeated Segments", rep_count)

                st.markdown("##### 📈 Frame-by-Frame Signal Trajectories")
                c_ch1, c_ch2 = st.columns(2)
                with c_ch1:
                    st.caption("Spectral Flux Trajectory")
                    st.line_chart(dict(time=tamp_res.time_points, flux=tamp_res.spectral_flux), x="time", y="flux")
                with c_ch2:
                    st.caption("RMS Energy & Noise Floor Trajectory")
                    st.line_chart(dict(time=tamp_res.time_points, rms=tamp_res.rms_energy, noise=tamp_res.noise_floor), x="time", y=["rms", "noise"])

                st.markdown("##### 📋 Timestamped Acoustic Findings")
                if tamp_res.findings:
                    f_rows = [
                        {
                            "Timestamp Window": f"{f.start_sec:.2f}s - {f.end_sec:.2f}s",
                            "Indicator": f.indicator,
                            "Measured Value": f"{f.value:.4f} {f.unit}",
                            "Threshold": f"{f.threshold:.4f}",
                            "Severity": f.severity,
                            "Flag": f"[{f.interpretation_flag}]",
                            "Description": f.description,
                        }
                        for f in tamp_res.findings
                    ]
                    render_styled_table(f_rows)
                else:
                    st.success("No sharp spectral or energy discontinuities detected above threshold.")

                st.caption(f"ℹ️ {tamp_res.disclaimer}")

    # -------------------------------------------------------------------------
    # TAB 3: Speaker Comparison Baseline
    # -------------------------------------------------------------------------
    with t_spk:
        st.markdown("#### 👥 Speaker Consistency & Acoustic Profile Comparison")
        st.error(
            "🛑 **CRITICAL FORENSIC LIMITATION & PROHIBITION**: Biometric speaker verification and individual "
            "identity confirmation are strictly UNAVAILABLE in this release. The dataset lacks cross-channel speaker "
            "identity ground-truth, and deep embedding networks (ECAPA-TDNN / x-vectors) are not installed. "
            "The comparison below evaluates exploratory acoustic spectral distance only. Asserting speaker identity "
            "from these metrics is scientifically invalid and legally impermissible."
        )

        other_evidence = [e for e in all_evidence if e["evidence_id"] != exhibit["evidence_id"]]
        if not other_evidence:
            st.warning("Case contains only one exhibit. Ingest a second audio file to perform acoustic comparison.")
        else:
            other_opts = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in other_evidence}
            comp_label = st.selectbox("Select Comparison Exhibit (Exhibit B)", options=list(other_opts.keys()), key="p7_spk_b")
            exhibit_b = other_opts[comp_label]

            if st.button("🔬 Compute Acoustic Distance Baseline", key="btn_run_spk"):
                with st.spinner("Extracting acoustic vectors and calculating spectral distance..."):
                    wf_a, sr_a = load_audio(stored_file, target_sr=16000, mono=True)
                    wf_b, sr_b = load_audio(Path(exhibit_b["stored_path"]), target_sr=16000, mono=True)

                    raw_a = extract_basic_features(wf_a, sample_rate=sr_a, n_mfcc=20)
                    raw_b = extract_basic_features(wf_b, sample_rate=sr_b, n_mfcc=20)
                    vec_a = align_feature_vector(raw_a, engine.feature_names)
                    vec_b = align_feature_vector(raw_b, engine.feature_names)

                    spk_res = compare_acoustic_profiles(
                        vec_a, vec_b, evidence_a_id=exhibit["evidence_id"], evidence_b_id=exhibit_b["evidence_id"]
                    )

                    c_s1, c_s2, c_s3 = st.columns(3)
                    c_s1.metric("Cosine Similarity", f"{spk_res.cosine_similarity:.4f}")
                    c_s2.metric("Euclidean Distance", f"{spk_res.euclidean_distance:.4f}")
                    c_s3.metric("Acoustic Divergence", f"{spk_res.acoustic_divergence_pct:.1f}%")

                    st.markdown("##### 🎚️ Sub-band Acoustic Similarity")
                    band_rows = [{"Acoustic Band": k, "Cosine Resemblance": f"{v:.3f}"} for k, v in spk_res.band_similarity.items()]
                    render_styled_table(band_rows)

                    st.markdown("##### ⚠️ Missing Prerequisites for Legal Biometric Verification")
                    for req in spk_res.prerequisites_missing:
                        st.markdown(f"- ❌ {req}")

    # -------------------------------------------------------------------------
    # TAB 4: Generator Characterization
    # -------------------------------------------------------------------------
    with t_gen:
        st.markdown("#### 🤖 Synthetic Generator & Vocoder Characterization")
        st.warning(
            "⚠️ **FORENSIC NOTICE**: Supervised multi-class generator attribution is UNAVAILABLE because training data "
            "lacks vocoder class labels. The analysis below measures physical spectral bandwidth, high-frequency cutoff, "
            "and rolloff characteristics to provide preliminary acoustic vocoder family hypotheses."
        )

        if st.button("🔍 Profile Vocoder Bandwidth & Artifacts", key="btn_run_gen"):
            with st.spinner("Analyzing spectral energy distribution and cutoff frequencies..."):
                wf, sr = load_audio(stored_file, target_sr=16000, mono=True)
                gen_res = profile_generator_artifacts(wf, sr, evidence_id=exhibit["evidence_id"])

                c_g1, c_g2, c_g3 = st.columns(3)
                c_g1.metric("Effective Bandwidth", f"{gen_res.effective_bandwidth_hz:,.0f} Hz")
                c_g2.metric("85% Spectral Rolloff", f"{gen_res.spectral_rolloff_85_hz:,.0f} Hz")
                c_g3.metric("High-Frequency Ratio (>7kHz)", f"{gen_res.high_freq_energy_ratio * 100:.2f}%")

                st.markdown(f"**Candidate Architecture Hypothesis**: `{gen_res.candidate_generator_class}`")
                st.caption(f"Status: **{gen_res.confidence_level}** (Requires independent acoustic confirmation)")

    # -------------------------------------------------------------------------
    # TAB 5: Acoustic Anomaly Detection
    # -------------------------------------------------------------------------
    with t_anom:
        st.markdown("#### 🚨 Unsupervised Frame-Level Acoustic Anomaly Detection")
        st.info("Uses Isolation Forest / Local Outlier Factor over multi-feature acoustic manifolds to flag statistical outliers.")

        c_a1, c_a2 = st.columns(2)
        anom_method = c_a1.selectbox("Outlier Detection Algorithm", ["IsolationForest", "LOF"], index=0, key="p7_anom_alg")
        anom_contam = c_a2.slider("Contamination Fraction", min_value=0.01, max_value=0.15, value=0.05, step=0.01, key="p7_anom_cont")

        if st.button("🚨 Run Anomaly Screening", key="btn_run_anom"):
            with st.spinner("Fitting unsupervised manifold and computing frame anomaly scores..."):
                wf, sr = load_audio(stored_file, target_sr=16000, mono=True)
                anom_res = detect_acoustic_anomalies(
                    wf, sr, evidence_id=exhibit["evidence_id"], contamination=anom_contam, method=anom_method
                )

                c_am1, c_am2, c_am3 = st.columns(3)
                c_am1.metric("Flagged Outlier Frames", anom_res.total_anomaly_frames)
                c_am2.metric("Outlier Fraction", f"{anom_res.outlier_fraction * 100:.2f}%")
                c_am3.metric("Anomaly Score Threshold", f"{anom_res.score_threshold:.4f}")

                st.caption("Frame-by-Frame Anomaly Score Trajectory")
                st.line_chart(dict(time=anom_res.time_points, anomaly_score=anom_res.anomaly_scores), x="time", y="anomaly_score")

                if anom_res.findings:
                    st.markdown("##### 🚩 Detected Acoustic Anomaly Clusters")
                    anom_rows = [
                        {
                            "Timestamp Window": f"{f.start_sec:.2f}s - {f.end_sec:.2f}s",
                            "Peak Anomaly Score": f"{f.value:.4f}",
                            "Threshold": f"{f.threshold:.4f}",
                            "Severity": f.severity,
                            "Description": f.description,
                        }
                        for f in anom_res.findings
                    ]
                    render_styled_table(anom_rows)
                else:
                    st.success("No statistical acoustic outliers detected.")

    # -------------------------------------------------------------------------
    # TAB 6: Advanced Signal Forensic Workspace
    # -------------------------------------------------------------------------
    with t_sig:
        st.markdown("#### 🔬 Advanced Signal Laboratory & Playback Inspection")
        wf, sr = load_audio(stored_file, target_sr=16000, mono=True)
        dur = len(wf) / sr

        st.audio(str(stored_file))
        st.markdown(f"**Duration**: `{dur:.2f}s` | **Sample Rate**: `{sr:,} Hz` | **Total Samples**: `{len(wf):,}`")

        # Interactive Cropper
        st.markdown("##### ✂️ Interactive Temporal Region Selector")
        c_cr1, c_cr2 = st.columns(2)
        start_crop = c_cr1.slider("Start Second", min_value=0.0, max_value=max(0.0, dur - 0.2), value=0.0, step=0.1)
        end_crop = c_cr2.slider("End Second", min_value=min(dur, start_crop + 0.1), max_value=dur, value=min(dur, start_crop + 2.0), step=0.1)

        idx_start = int(start_crop * sr)
        idx_end = int(end_crop * sr)
        sub_wf = wf[idx_start:idx_end]

        if len(sub_wf) > 0:
            st.caption(f"Inspecting window [{start_crop:.2f}s - {end_crop:.2f}s] ({len(sub_wf):,} samples)")
            # Decimate for chart
            step_sz = max(1, len(sub_wf) // 600)
            dec_time = np.linspace(start_crop, end_crop, len(sub_wf[::step_sz]))
            dec_wave = sub_wf[::step_sz]
            st.line_chart(dict(time=dec_time, amplitude=dec_wave), x="time", y="amplitude")

    # -------------------------------------------------------------------------
    # TAB 7: Multi-Evidence Case Correlation
    # -------------------------------------------------------------------------
    with t_corr:
        st.markdown("#### 🔗 Cross-Exhibit Case Correlation & Alignment")
        st.caption("Correlates multiple exhibits in the same case for cryptographic identity, acoustic similarity, and temporal offset.")

        other_evidence = [e for e in all_evidence if e["evidence_id"] != exhibit["evidence_id"]]
        if not other_evidence:
            st.warning("Only one exhibit exists. Upload an additional exhibit to correlate.")
        else:
            other_opts = {f"[{e['case_id']}] {e['original_filename']} ({e['evidence_id']})": e for e in other_evidence}
            target_ev_label = st.selectbox("Select Exhibit to Correlate (Exhibit B)", options=list(other_opts.keys()), key="p7_corr_b")
            exhibit_b = other_opts[target_ev_label]

            if st.button("🔗 Execute Multi-Evidence Correlation", key="btn_run_corr"):
                with st.spinner("Performing cross-exhibit hash comparison, temporal offset search, and acoustic correlation..."):
                    wf_a, sr_a = load_audio(stored_file, target_sr=16000, mono=True)
                    wf_b, sr_b = load_audio(Path(exhibit_b["stored_path"]), target_sr=16000, mono=True)

                    raw_a = extract_basic_features(wf_a, sample_rate=sr_a, n_mfcc=20)
                    raw_b = extract_basic_features(wf_b, sample_rate=sr_b, n_mfcc=20)
                    vec_a = align_feature_vector(raw_a, engine.feature_names)
                    vec_b = align_feature_vector(raw_b, engine.feature_names)

                    ev_model_a = ForensicEvidence(**{k: v for k, v in exhibit.items() if k != "id"})
                    ev_model_b = ForensicEvidence(**{k: v for k, v in exhibit_b.items() if k != "id"})

                    corr_rep = correlate_exhibits(
                        ev_model_a, ev_model_b, audio_a=wf_a, audio_b=wf_b, features_a=vec_a, features_b=vec_b, db=db
                    )

                    c_c1, c_c2, c_c3 = st.columns(3)
                    sha_badge = "✅ IDENTICAL (Bit-Exact)" if corr_rep.sha256_match else "❌ DISTINCT"
                    c_c1.markdown(f"**SHA-256 Identity**: {sha_badge}")
                    c_c2.metric("Overall Correlation Score", f"{corr_rep.correlation_score * 100:.1f}%")
                    c_c3.metric("Temporal Alignment Offset", f"{corr_rep.temporal_alignment.get('offset_seconds', 0.0):+.3f}s")

                    st.markdown("##### 📋 Metadata Differences")
                    m_rows = [
                        {"Attribute": "Original Filenames", "Exhibit A": corr_rep.filename_a, "Exhibit B": corr_rep.filename_b},
                        {"Attribute": "Duration (seconds)", "Exhibit A": f"{exhibit['duration_seconds']:.2f}s", "Exhibit B": f"{exhibit_b['duration_seconds']:.2f}s"},
                        {"Attribute": "Sample Rate", "Exhibit A": f"{exhibit['sample_rate']:,} Hz", "Exhibit B": f"{exhibit_b['sample_rate']:,} Hz"},
                        {"Attribute": "Container Format", "Exhibit A": exhibit['container_format'], "Exhibit B": exhibit_b['container_format']},
                    ]
                    render_styled_table(m_rows)

                    if corr_rep.findings:
                        st.markdown("##### 🚩 Correlation Findings")
                        f_corr_rows = [
                            {
                                "Finding ID": f.finding_id,
                                "Indicator": f.indicator,
                                "Value": f"{f.value:.4f}",
                                "Severity": f.severity,
                                "Description": f.description,
                            }
                            for f in corr_rep.findings
                        ]
                        render_styled_table(f_corr_rows)

    # -------------------------------------------------------------------------
    # TAB 8: Phase 7 Reports
    # -------------------------------------------------------------------------
    with t_p7_docs:
        st.markdown("#### 📥 Phase 7 Forensic Intelligence & Explainability Reports")
        p7_reports = [
            ("PHASE_7_PRE_IMPLEMENTATION_AUDIT.md", "Phase 7.0 Pre-Implementation Audit"),
            ("PHASE_7_EXPLAINABLE_AI_REPORT.md", "Phase 7.1 Explainable AI & Attribution"),
            ("PHASE_7_AUDIO_TAMPERING_ANALYSIS_REPORT.md", "Phase 7.2 Signal Manipulation & Splicing"),
            ("PHASE_7_SPEAKER_COMPARISON_REPORT.md", "Phase 7.3 Speaker Consistency & Comparison"),
            ("PHASE_7_SYNTHETIC_GENERATOR_ANALYSIS_REPORT.md", "Phase 7.4 Generator & Vocoder Profiling"),
            ("PHASE_7_ACOUSTIC_ANOMALY_REPORT.md", "Phase 7.5 Acoustic Anomaly Detection"),
            ("PHASE_7_SIGNAL_ANALYSIS_REPORT.md", "Phase 7.6 Advanced Signal Workspace"),
            ("PHASE_7_EVIDENCE_CORRELATION_REPORT.md", "Phase 7.7 Multi-Evidence Case Correlation"),
            ("PHASE_7_DASHBOARD_INTEGRATION_REPORT.md", "Phase 7.8 Dashboard Integration Architecture"),
            ("PHASE_7_IMPLEMENTATION_REPORT.md", "Phase 7.9 Implementation Deliverable"),
            ("PHASE_7_TESTING_REPORT.md", "Phase 7.10 Automated Testing & Validation"),
            ("PHASE_7_USER_GUIDE.md", "Phase 7.11 Investigator Operating Guide"),
            ("PHASE_7_SCIENTIFIC_LIMITATIONS_REPORT.md", "Phase 7.12 Forensic Scientific Limitations"),
            ("PHASE_7_FINAL_CONSOLIDATED_REPORT.md", "Phase 7.13 Master Consolidated Summary"),
        ]

        c_col1, c_col2 = st.columns(2)
        for idx, (fname, desc) in enumerate(p7_reports):
            fpath = root / fname
            target_col = c_col1 if idx % 2 == 0 else c_col2
            with target_col:
                if fpath.is_file():
                    target_col.download_button(
                        f"📄 {desc}",
                        data=fpath.read_text(encoding="utf-8"),
                        file_name=fname,
                        mime="text/markdown",
                        key=f"dl_p7_{idx}",
                    )
                else:
                    target_col.info(f"{fname} (Pending generation)")


# =============================================================================
# 11. System Settings & Diagnostics
# =============================================================================
def render_settings_and_diagnostics():
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">System Settings & Diagnostics</div>
            <div class="lab-subtitle">
                Runtime environment verification, model schema inspection, and OOD threshold calibration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("#### 💻 Runtime Environment")
        env = get_environment_metadata()
        st.json(env)

    with c_d2:
        st.markdown("#### 📏 Empirical OOD Threshold Bounds")
        stats_path = root / "models" / "training_distribution_stats.json"
        if stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            st.json(stats.get("ood_thresholds", {}))
        else:
            st.info("OOD statistics file not found.")

    st.markdown("---")
    st.markdown("#### 🗄️ Database Integrity Status")
    db_file = db.db_path
    st.markdown(f"Database Location: `{db_file}` (Size: {db_file.stat().st_size:,} bytes)")
    if st.button("Run SQLite Integrity Check"):
        with sqlite3.connect(db_file) as conn:
            res = conn.execute("PRAGMA integrity_check;").fetchall()
            st.success(f"SQLite PRAGMA integrity_check: {res[0][0]}")


if __name__ == "__main__":
    main()
