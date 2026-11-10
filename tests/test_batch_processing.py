"""Unit and integration tests for batch evidence processing and case management."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.batch_processing import process_audio_batch, generate_master_case_pdf
from src.utils import project_root


@pytest.fixture
def temp_evidence_dir(tmp_path: Path):
    """Create a temporary directory containing mock forensic audio exhibits."""
    case_dir = tmp_path / "case_exhibits"
    case_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Exhibit 1: Tone A
    wave1 = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(str(case_dir / "exhibit_alpha.wav"), wave1, sr)

    # Exhibit 2: Tone B
    wave2 = (0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    sf.write(str(case_dir / "exhibit_beta.wav"), wave2, sr)

    return case_dir


def test_batch_processing_complete_run(temp_evidence_dir, tmp_path):
    out_dir = tmp_path / "case_reports"
    progress_logs = []

    def on_progress(curr, tot, fname):
        progress_logs.append((curr, tot, fname))

    case_summary = process_audio_batch(
        temp_evidence_dir,
        case_id="CASE-UNIT-TEST",
        examiner_name="Special Agent Analyst",
        model_name="ensemble",
        output_dir=out_dir,
        generate_visuals=False,
        generate_individual_reports=True,
        progress_callback=on_progress,
    )

    # Verify statistics
    stats = case_summary["statistics"]
    assert stats["total_exhibits"] == 2
    assert stats["total_duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert len(progress_logs) == 2

    # Verify generated report files
    csv_ledger = Path(case_summary["reports"]["csv_ledger"])
    md_report = Path(case_summary["reports"]["markdown_report"])
    pdf_report = Path(case_summary["reports"]["pdf_report"])

    assert csv_ledger.is_file()
    assert md_report.is_file()
    assert pdf_report.is_file()

    # Verify CSV ledger content
    with csv_ledger.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["file_name"] in {"exhibit_alpha.wav", "exhibit_beta.wav"}
        assert len(reader[0]["file_sha256"]) == 64

    # Verify Markdown report content
    md_text = md_report.read_text(encoding="utf-8")
    assert "MASTER FORENSIC CASE INVESTIGATION REPORT" in md_text
    assert "CASE-UNIT-TEST" in md_text
    assert "exhibit_alpha.wav" in md_text
    assert "exhibit_beta.wav" in md_text

    # Verify PDF non-empty
    assert pdf_report.stat().st_size > 1000


def test_batch_processing_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="No valid audio exhibits found"):
        process_audio_batch(empty_dir)


def test_master_case_pdf_generation(tmp_path):
    mock_case = {
        "case_id": "CASE-PDF-TEST-001",
        "examiner_name": "Senior Forensic Scientist",
        "timestamp_utc": "2026-11-10T12:00:00Z",
        "primary_model_name": "Calibrated Ensemble",
        "statistics": {
            "total_exhibits": 2,
            "genuine_count": 1,
            "synthetic_count": 1,
            "indeterminate_count": 0,
            "total_duration_seconds": 120.0,
            "mean_duration_seconds": 60.0,
        },
        "exhibits": [
            {
                "file_name": "recorded_interview.wav",
                "file_sha256": "1111222233334444555566667777888899990000111122223333444455556666",
                "predicted_label": "GENUINE",
                "calibrated_probability_synthetic": 0.02,
                "duration_seconds": 60.0,
                "status": "COMPLETED",
            },
            {
                "file_name": "voicemail_threat.mp3",
                "file_sha256": "aaaa9999bbbb8888cccc7777dddd6666eeee5555ffff44440000111122223333",
                "predicted_label": "SYNTHETIC",
                "calibrated_probability_synthetic": 0.98,
                "duration_seconds": 60.0,
                "status": "COMPLETED",
            },
        ],
    }

    pdf_out = tmp_path / "master_report.pdf"
    res_path = generate_master_case_pdf(mock_case, pdf_out)
    assert res_path.is_file()
    assert res_path.stat().st_size > 1500

