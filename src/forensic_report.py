"""Forensic report generation adhering to ISO/IEC 27037 digital evidence principles.

Generates formal forensic analysis reports for legal and investigative proceedings:
- Documents digital evidence provenance and chain of custody.
- Details signal preprocessing and acoustic feature extraction methodology.
- Presents classification findings and posterior probability estimates.
- Articulates methodological limitations and mandatory evidentiary disclaimers.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils import project_root

logger = logging.getLogger(__name__)


def create_forensic_report(
    prediction_result: Dict[str, Any],
    case_number: Optional[str] = None,
    examiner_name: str = "Automated Audio Forensics Unit",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a formal markdown forensic examination report from prediction findings.

    Parameters
    ----------
    prediction_result : Dict[str, Any]
        Output dictionary from ``src.audio_prediction.predict_audio``.
    case_number : str, optional
        Investigative or court case reference identifier.
    examiner_name : str, default="Automated Audio Forensics Unit"
        Name or title of the examining agent/analyst.
    output_path : Union[str, Path], optional
        Destination filepath for the report. If omitted, defaults to
        ``results/reports/FORENSIC_EXAMINATION_REPORT_<filename>.md``.

    Returns
    -------
    str
        Complete markdown report string.
    """
    root = project_root()
    case_id = case_number or f"CASE-{datetime.date.today().strftime('%Y%m%d')}-001"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    file_name = prediction_result.get("file_name", "UNKNOWN_FILE")
    file_sha256 = prediction_result.get("file_sha256", "UNKNOWN_HASH")
    file_size = prediction_result.get("file_size_bytes", 0)
    orig_fmt = prediction_result.get("original_format", "N/A")
    orig_sr = prediction_result.get("original_sample_rate", "N/A")
    duration = prediction_result.get("duration_seconds", 0.0)
    pred_label = prediction_result.get("predicted_label", "UNKNOWN")
    confidence = prediction_result.get("confidence", 0.0)
    prob_synth = prediction_result.get("probability_synthetic")
    prob_real = prediction_result.get("probability_genuine")
    model_name = prediction_result.get("model_name", "Forensic Machine Learning Pipeline")
    assessment = prediction_result.get("forensic_assessment", "")

    synth_str = f"{prob_synth:.2%}" if prob_synth is not None else "N/A"
    real_str = f"{prob_real:.2%}" if prob_real is not None else "N/A"

    report_md = f"""# FORENSIC AUDIO EXAMINATION REPORT

**Case Identifier:** `{case_id}`  
**Examination Date:** {timestamp}  
**Examiner:** {examiner_name}  
**Laboratory:** AI Audio Forensics Investigation Unit  
**Status:** Completed Examination  

---

## 1. Evidence Description & Chain of Custody

The digital audio exhibit described below was received for acoustic authenticity evaluation:

| Attribute | Specification |
|---|---|
| **Exhibit Filename** | `{file_name}` |
| **Evidence File Size** | {file_size:,} bytes |
| **Cryptographic Hash (SHA-256)** | `{file_sha256}` |
| **Container Format** | {orig_fmt} |
| **Original Sampling Rate** | {orig_sr} Hz |
| **Audio Duration** | {duration:.2f} seconds |

*Verification Note: Cryptographic SHA-256 hash was generated upon intake to ensure evidence integrity throughout processing.*

---

## 2. Signal Preprocessing & Feature Extraction Methodology

The audio exhibit was analyzed using standardized, reproducible signal processing transforms:

1. **Standardisation:** Downmixed to single-channel (mono) floating-point representation and resampled to 16,000 Hz. Peak amplitude normalized to 1.0 while preserving silence.
2. **Feature Extraction:** Evaluated across 9 standard acoustic and spectral dimensions:
   - 20 Mel-Frequency Cepstral Coefficients (MFCCs)
   - Spectral Centroid (spectral brightness / center of mass)
   - Spectral Bandwidth (frequency spread)
   - Spectral Rolloff (85% cumulative energy threshold)
   - Zero-Crossing Rate (time-domain frequency indicator)
   - Root-Mean-Square (RMS) Energy (amplitude dynamics)
   - Chroma Feature Representation (12 semitone pitch classes)
   - Spectral Contrast (peak-to-valley energy difference across octave sub-bands)
   - Spectral Flatness (Wiener entropy / tonality vs. noisiness)
3. **Classification Model:** Classified using trained `{model_name}`.

---

## 3. Examination Findings & Results

Based on statistical acoustic evaluation and machine learning inference, the findings are:

- **Classification Finding:** **{pred_label}**
- **Decision Confidence:** **{confidence:.2%}**
- **Posterior Probability (Synthetic Deepfake):** **{synth_str}**
- **Posterior Probability (Genuine Speech):** **{real_str}**

### Forensic Assessment Statement:
> {assessment}

---

## 4. Evidentiary Limitations & Technical Boundaries

Forensic voice analysis is inherently bounded by acoustic and acquisition conditions:
1. **Transcoding & Codec Compression:** MP3, AAC, and telephony codecs introduce spectral quantization artifacts that can mimic or obscure vocoder synthesis cues.
2. **Channel Variability:** Varying acoustic environments, microphone responses, and room reverberation may alter high-frequency spectral rolloff.
3. **Out-of-Domain Voice Generators:** Performance was calibrated on speech synthesis and voice conversion systems; novel zero-shot generative models may exhibit varying anomaly patterns.

---

## 5. Mandatory Forensic Disclaimer

> **IMPORTANT LEGAL NOTICE:**  
> This examination report provides an investigative intelligence screening assessment based on automated acoustic signal analysis. In accordance with digital forensics principles (referencing ISO/IEC 27037 guidelines and SWGDE recommendations), this finding does **not** constitute definitive legal proof of authentic human origin or criminal forgery. It must be evaluated in conjunction with corroborating investigative evidence, witness testimony, and verified chain of custody.

---
*End of Formal Report.*
"""

    if output_path is None:
        rep_dir = root / "results" / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        out_file = rep_dir / f"FORENSIC_EXAMINATION_REPORT_{Path(file_name).stem}.md"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    out_file.write_text(report_md, encoding="utf-8")
    logger.info("Saved forensic report to %s", out_file)
    return report_md
