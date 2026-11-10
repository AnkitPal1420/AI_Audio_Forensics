"""Recording-level manifest and group-aware slice dataset generation.

Inspects every audio exhibit, extracts metadata, tracks source-to-conversion
relationships, and generates an auditable, group-labeled 1-second slice dataset.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.audio_preprocessing import compute_file_hash
from src.feature_extraction import CANONICAL_FEATURE_NAMES
from src.utils import project_root

logger = logging.getLogger(__name__)

# Known verified subjects in the DEEP-VOICE dataset
KNOWN_SUBJECTS = {
    "biden": "Joe Biden",
    "linus": "Linus Sebastian",
    "margot": "Margot Robbie",
    "musk": "Elon Musk",
    "obama": "Barack Obama",
    "ryan": "Ryan Gosling",
    "taylor": "Taylor Swift",
    "trump": "Donald Trump",
}


def parse_recording_metadata(filename: str, label_type: str) -> Dict[str, Any]:
    """Parse speaker, source, and target relationships from an exhibit filename."""
    stem = Path(filename).stem
    is_demo = "DEMO" in stem

    if label_type == "real":
        # e.g., biden-original, linus-original-DEMO
        parts = stem.replace("-DEMO", "").replace("_16k_mono", "").split("-")
        speaker_key = parts[0].lower()
        speaker_id = speaker_key if speaker_key in KNOWN_SUBJECTS else "unknown"
        return {
            "recording_id": stem,
            "speaker_id": speaker_id,
            "speaker_name": KNOWN_SUBJECTS.get(speaker_id, "Unknown"),
            "acoustic_source_speaker": speaker_id,
            "target_voice_profile": speaker_id,
            "conversion_pair": "none",
            "is_demonstration": is_demo,
            "label": "genuine",
            "label_idx": 0,
            "generation_method": "bona_fide_human_speech",
            "related_genuine_recording": stem,
            "group_id": speaker_id,
            "dataset_provenance": "DEEP-VOICE (Kaggle Benchmark)",
        }
    else:
        # e.g., linus-to-musk, Obama-to-Biden, linus-to-musk-DEMO
        clean_stem = stem.replace("-DEMO", "").replace("_16k_mono", "")
        if "-to-" in clean_stem:
            parts = clean_stem.split("-to-")
            src_key = parts[0].lower()
            tgt_key = parts[1].lower()
        else:
            src_key = "unknown"
            tgt_key = "unknown"

        src_id = src_key if src_key in KNOWN_SUBJECTS else "unknown"
        tgt_id = tgt_key if tgt_key in KNOWN_SUBJECTS else "unknown"

        return {
            "recording_id": stem,
            "speaker_id": tgt_id,  # Target voice timbre
            "speaker_name": KNOWN_SUBJECTS.get(tgt_id, "Unknown"),
            "acoustic_source_speaker": src_id,  # Underlying linguistic and acoustic source
            "target_voice_profile": tgt_id,
            "conversion_pair": f"{src_id}-to-{tgt_id}",
            "is_demonstration": is_demo,
            "label": "synthetic",
            "label_idx": 1,
            "generation_method": "RVC_v2 (Retrieval-based Voice Conversion)",
            "related_genuine_recording": f"{src_id}-original",
            "group_id": src_id,  # All conversions of the same source belong to the same cluster
            "dataset_provenance": "DEEP-VOICE (Kaggle Benchmark)",
        }


def build_recording_level_manifest(
    preprocessed_dir: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive recording-level manifest with SHA-256 hashes and speaker metadata."""
    root = project_root()
    p_dir = Path(preprocessed_dir) if preprocessed_dir else root / "dataset" / "preprocessed"
    out_file = Path(output_json) if output_json else root / "dataset" / "recording_level_manifest.json"

    manifest_entries: List[Dict[str, Any]] = []

    for category, label_type in [("real", "real"), ("fake", "fake")]:
        folder = p_dir / category
        if not folder.is_dir():
            continue

        for fpath in sorted(folder.glob("*.wav")):
            info = sf.info(str(fpath))
            meta = parse_recording_metadata(fpath.name, label_type)
            meta["file_path"] = str(fpath.resolve())
            meta["file_name"] = fpath.name
            meta["file_sha256"] = compute_file_hash(fpath)
            meta["duration_seconds"] = float(info.duration)
            meta["sample_rate"] = int(info.samplerate)
            meta["channels"] = int(info.channels)
            manifest_entries.append(meta)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    logger.info("Saved recording manifest with %d exhibits to %s", len(manifest_entries), out_file)
    return {
        "total_recordings": len(manifest_entries),
        "manifest_path": str(out_file),
        "entries": manifest_entries,
    }


def generate_group_aware_slice_dataset(
    manifest_entries: List[Dict[str, Any]],
    output_csv: Optional[Union[str, Path]] = None,
    max_fake_slices_per_recording: int = 65,
    sample_rate: int = 16_000,
) -> str:
    """Generate a balanced 1-second slice dataset with verified group IDs and standardized features."""
    root = project_root()
    out_path = Path(output_csv) if output_csv else root / "dataset" / "preprocessed_slices_dataset.csv"

    n_fft = 2048
    hop = 512
    n_mels = 128
    mel_pts = np.linspace(0.0, 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0), n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    mel_filters = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        low, mid, high = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (freqs - low) / (mid - low)
        falling = (high - freqs) / (high - mid)
        mel_filters[i] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (high - low))

    coeffs = np.arange(n_mels, dtype=np.float32)[:, None]
    bands = np.arange(n_mels, dtype=np.float32)[None, :]
    dct_basis = np.cos(np.pi / n_mels * (bands + 0.5) * coeffs)
    dct_basis[0] *= np.sqrt(1.0 / n_mels)
    dct_basis[1:] *= np.sqrt(2.0 / n_mels)

    chroma_filters = np.zeros((12, freqs.size), dtype=np.float32)
    valid_mask = freqs > 60.0
    if np.any(valid_mask):
        midi = 12.0 * np.log2(freqs[valid_mask] / 440.0) + 69.0
        pc = np.mod(np.round(midi), 12).astype(int)
        for p in range(12):
            chroma_filters[p, np.where(valid_mask)[0][pc == p]] = 1.0
        r_sums = chroma_filters.sum(axis=1, keepdims=True)
        chroma_filters = np.divide(chroma_filters, r_sums, out=chroma_filters, where=r_sums > 0)

    rows: List[Dict[str, Any]] = []

    for rec in manifest_entries:
        fpath = Path(rec["file_path"])
        if not fpath.is_file():
            continue

        waveform, sr = sf.read(str(fpath))
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Pad and frame
        pad_len = n_fft // 2
        padded = np.pad(waveform, (pad_len, pad_len), mode="reflect")
        num_frames = max(1, (padded.size - n_fft) // hop + 1)
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(num_frames, n_fft),
            strides=(padded.strides[0] * hop, padded.strides[0]),
        )
        window = np.hanning(n_fft).astype(np.float32)
        spectrum = np.fft.rfft(frames * window, axis=1).T
        magnitude = np.abs(spectrum)
        power = magnitude**2

        # Frame-level metrics
        rms_f = np.sqrt(np.mean(frames**2, axis=1))
        signs = np.sign(frames)
        signs[signs == 0] = 1
        zcr_f = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

        f_energy = np.sum(magnitude, axis=0)
        cent_f = np.divide(freqs @ magnitude, f_energy, out=np.zeros_like(f_energy), where=f_energy > 0)

        f_diff = freqs[:, None] - cent_f[None, :]
        bw_f = np.sqrt(np.divide(np.sum((f_diff**2) * magnitude, axis=0), f_energy, out=np.zeros_like(f_energy), where=f_energy > 0))

        cum_e = np.cumsum(magnitude, axis=0)
        thresh = 0.85 * f_energy
        ro_idx = np.argmax(cum_e >= thresh[None, :], axis=0)
        ro_f = freqs[ro_idx]

        mel_s = mel_filters @ power
        db_m = 10.0 * np.log10(np.maximum(1e-10, mel_s))
        db_m = np.maximum(db_m, np.max(db_m) - 80.0)
        mfcc_f = (dct_basis @ db_m)[:20]

        raw_c = chroma_filters @ power
        c_norm = np.linalg.norm(raw_c, axis=0, keepdims=True)
        chr_f = np.divide(raw_c, c_norm, out=raw_c, where=c_norm > 0)
        chr_mean_f = np.mean(chr_f, axis=0)

        frames_per_sec = int(round(sample_rate / hop))  # ~31 frames
        total_seconds = int(np.floor(waveform.size / sample_rate))

        if rec["label"] == "synthetic" and total_seconds > max_fake_slices_per_recording:
            # Subsample uniformly across the duration
            sec_indices = np.linspace(0, total_seconds - 1, max_fake_slices_per_recording, dtype=int)
        else:
            sec_indices = np.arange(total_seconds)

        for s_idx, sec in enumerate(sec_indices):
            f_start = sec * frames_per_sec
            f_end = min(f_start + frames_per_sec, num_frames)
            if f_end <= f_start:
                continue

            slice_row: Dict[str, Any] = {
                "slice_id": f"{rec['recording_id']}_slice_{s_idx:04d}",
                "recording_id": rec["recording_id"],
                "group_id": rec["group_id"],
                "speaker_id": rec["speaker_id"],
                "acoustic_source_speaker": rec["acoustic_source_speaker"],
                "label": rec["label_idx"],
                "slice_start_sec": float(sec),
                "slice_end_sec": float(sec + 1),
                "chroma_stft": float(np.mean(chr_mean_f[f_start:f_end])),
                "rms": float(np.mean(rms_f[f_start:f_end])),
                "spectral_centroid": float(np.mean(cent_f[f_start:f_end])),
                "spectral_bandwidth": float(np.mean(bw_f[f_start:f_end])),
                "rolloff": float(np.mean(ro_f[f_start:f_end])),
                "zero_crossing_rate": float(np.mean(zcr_f[f_start:f_end])),
            }
            for m in range(20):
                slice_row[f"mfcc{m+1}"] = float(np.mean(mfcc_f[m, f_start:f_end]))

            rows.append(slice_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved group-aware slice dataset with %d slices to %s", len(rows), out_path)
    return str(out_path)
