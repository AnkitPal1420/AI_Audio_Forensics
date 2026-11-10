"""Audio loading, validation, hashing, and preprocessing pipeline.

This module provides reproducible, forensically sound audio preprocessing:
- Preserves original evidence files unchanged.
- Computes cryptographic SHA-256 hashes for chain of custody.
- Validates file integrity, rejecting empty, corrupted, or unsupported files safely.
- Standardises audio to mono floating-point samples at a specified sample rate (default 16 kHz).
- Peak-normalises signals while preserving silence.
- Generates comprehensive dataset manifests and summaries.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:  # pragma: no cover
    soxr = None

logger = logging.getLogger(__name__)

AudioPath = Union[str, Path]
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioProcessingError(ValueError):
    """Base exception for audio loading and preprocessing failures."""


class CorruptedAudioError(AudioProcessingError):
    """Raised when an audio file header or payload cannot be parsed."""


class EmptyAudioError(AudioProcessingError):
    """Raised when an audio file is 0 bytes or contains zero audio samples."""


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when an audio file extension or encoding is unsupported."""


def compute_file_hash(path: AudioPath, algorithm: str = "sha256") -> str:
    """Compute the cryptographic hash of an audio file for evidence integrity.

    Reads the file in binary chunks so that large evidence files can be hashed
    without loading their entire contents into memory.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_audio_metadata(path: AudioPath) -> Dict[str, Any]:
    """Inspect an audio file and extract file and encoding metadata safely.

    Does not decode the full audio into memory. If the file is unreadable,
    empty, or corrupted, the returned dictionary notes the failure status
    without raising an unhandled exception.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "exists": False,
            "status": "missing",
            "error": "File does not exist",
        }

    file_size = file_path.stat().st_size
    if file_size == 0:
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": 0,
            "exists": True,
            "status": "empty",
            "error": "File size is 0 bytes",
        }

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": file_size,
            "exists": True,
            "format": suffix.lstrip(".").upper(),
            "status": "unsupported",
            "error": f"Unsupported format: {suffix}",
        }

    try:
        info = sf.info(str(file_path))
        file_hash = compute_file_hash(file_path)
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "format": info.format,
            "subtype": info.subtype,
            "channels": info.channels,
            "sample_rate": info.samplerate,
            "duration_seconds": float(info.duration),
            "frames": info.frames,
            "exists": True,
            "status": "valid",
            "error": None,
        }
    except Exception as exc:
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": file_size,
            "exists": True,
            "format": suffix.lstrip(".").upper(),
            "status": "corrupted",
            "error": str(exc),
        }


def convert_to_mono(waveform: np.ndarray) -> np.ndarray:
    """Downmix a multi-channel waveform to a 1D mono signal by channel averaging."""
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        # Channels might be along axis 0 or axis 1; soundfile outputs (frames, channels)
        if samples.shape[1] <= 8:  # (frames, channels)
            return np.mean(samples, axis=1, dtype=np.float32)
        elif samples.shape[0] <= 8:  # (channels, frames)
            return np.mean(samples, axis=0, dtype=np.float32)
        else:
            return np.mean(samples, axis=-1, dtype=np.float32)
    raise ValueError("waveform must be 1D mono or 2D multi-channel")


def resample_waveform(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample a 1D mono waveform from orig_sr to target_sr deterministically."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers")
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    if soxr is not None:
        resampled = soxr.resample(waveform.astype(np.float32), orig_sr, target_sr)
        return resampled.astype(np.float32, copy=False)

    # Fallback linear interpolation when soxr is unavailable
    duration = waveform.size / orig_sr
    target_length = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, waveform.size, endpoint=False)
    target_times = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(target_times, orig_times, waveform).astype(np.float32)


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize a waveform while leaving silence unchanged."""
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("waveform must be a one-dimensional mono signal")
    if not np.all(np.isfinite(samples)):
        raise ValueError("waveform must contain only finite values")

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples.copy()
    return samples / peak


def load_audio(
    path: AudioPath,
    sample_rate: int = 16_000,
    mono: bool = True,
    normalize: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, standardising to mono and target ``sample_rate``.

    Supports WAV, MP3, and FLAC containers without modifying the source file.
    Safely rejects missing, empty, corrupted, and unsupported files with clear
    exceptions.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    if audio_path.stat().st_size == 0:
        raise EmptyAudioError(f"Audio file contains 0 bytes: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    try:
        raw_waveform, loaded_sr = sf.read(str(audio_path), dtype="float32")
    except Exception as exc:
        raise CorruptedAudioError(f"Failed to decode audio file {audio_path}: {exc}") from exc

    if raw_waveform.size == 0:
        raise EmptyAudioError(f"Audio file contains no samples: {audio_path}")

    waveform = convert_to_mono(raw_waveform) if mono else raw_waveform
    if not np.all(np.isfinite(waveform)):
        raise CorruptedAudioError(f"Audio file contains non-finite values (NaN/Inf): {audio_path}")

    if sample_rate != loaded_sr:
        waveform = resample_waveform(waveform, orig_sr=loaded_sr, target_sr=sample_rate)

    if max_duration is not None and max_duration > 0:
        max_samples = int(round(max_duration * sample_rate))
        waveform = waveform[:max_samples]

    if normalize:
        waveform = normalize_audio(waveform)

    return waveform.astype(np.float32, copy=False), sample_rate


def standardize_audio(
    waveform: np.ndarray,
    source_sr: int,
    target_sr: int = 16_000,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Standardise an in-memory waveform to mono, target_sr, and normalised amplitude."""
    mono_wave = convert_to_mono(waveform)
    resampled = resample_waveform(mono_wave, orig_sr=source_sr, target_sr=target_sr)
    if normalize:
        resampled = normalize_audio(resampled)
    return resampled, target_sr


def preprocess_and_save(
    input_path: AudioPath,
    output_path: AudioPath,
    target_sr: int = 16_000,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Preprocess an original evidence file and save a standardised derivative.

    The original evidence file is NEVER modified or overwritten. The output is
    written as a standardised 16 kHz mono WAV file, and both source and
    derivative SHA-256 hashes are recorded for forensic auditability.
    """
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()
    if in_file == out_file:
        raise ValueError("Derivative output path cannot overwrite original evidence file")

    metadata = get_audio_metadata(in_file)
    if metadata.get("status") != "valid":
        raise AudioProcessingError(
            f"Cannot preprocess invalid file {in_file}: {metadata.get('error')}"
        )

    waveform, sr = load_audio(in_file, sample_rate=target_sr, mono=True, normalize=normalize)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), waveform, sr, subtype="PCM_16")

    derivative_hash = compute_file_hash(out_file)
    return {
        "source_file": str(in_file),
        "source_sha256": metadata["sha256"],
        "source_sample_rate": metadata["sample_rate"],
        "source_channels": metadata["channels"],
        "source_duration_seconds": metadata["duration_seconds"],
        "derivative_file": str(out_file),
        "derivative_sha256": derivative_hash,
        "derivative_sample_rate": sr,
        "derivative_channels": 1,
        "derivative_duration_seconds": float(waveform.size / sr),
        "status": "success",
    }


def preprocess_dataset(
    dataset_dir: Optional[AudioPath] = None,
    output_dir: Optional[AudioPath] = None,
    target_sr: int = 16_000,
    normalize: bool = True,
    manifest_path: Optional[AudioPath] = None,
) -> Dict[str, Any]:
    """Execute the standardisation pipeline on the dataset and build a manifest.

    Scans genuine ('real') and synthetic ('fake') audio recordings, standardises
    each to mono 16 kHz WAV, stores derivatives in ``output_dir``, and generates
    a comprehensive forensic manifest with SHA-256 evidence tracking.
    """
    from src.utils import project_root

    root = project_root()
    base_data = Path(dataset_dir) if dataset_dir else root / "dataset"
    out_base = Path(output_dir) if output_dir else base_data / "preprocessed"
    manifest_file = Path(manifest_path) if manifest_path else base_data / "evidence_manifest.json"

    classes = {"real": "genuine", "fake": "synthetic"}
    records: List[Dict[str, Any]] = []
    missing_or_corrupted: List[Dict[str, Any]] = []
    sample_rates: Dict[int, int] = {}
    durations: List[float] = []

    for sub_dir, class_label in classes.items():
        src_dir = base_data / sub_dir
        dest_dir = out_base / sub_dir
        if not src_dir.is_dir():
            continue

        for audio_file in sorted(src_dir.iterdir()):
            if not audio_file.is_file():
                continue
            if audio_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            meta = get_audio_metadata(audio_file)
            if meta.get("status") != "valid":
                missing_or_corrupted.append({
                    "file_path": str(audio_file),
                    "file_name": audio_file.name,
                    "class_label": class_label,
                    "status": meta.get("status"),
                    "error": meta.get("error"),
                })
                continue

            sr = meta["sample_rate"]
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            durations.append(meta["duration_seconds"])

            # Create standardised derivative
            dest_file = dest_dir / f"{audio_file.stem}_16k_mono.wav"
            prep_info = preprocess_and_save(
                input_path=audio_file,
                output_path=dest_file,
                target_sr=target_sr,
                normalize=normalize,
            )
            prep_info["class_label"] = class_label
            prep_info["category"] = sub_dir
            records.append(prep_info)

    dur_array = np.array(durations, dtype=np.float64) if durations else np.array([], dtype=np.float64)
    duration_stats = {
        "count": int(dur_array.size),
        "total_seconds": float(np.sum(dur_array)) if dur_array.size else 0.0,
        "total_hours": float(np.sum(dur_array) / 3600.0) if dur_array.size else 0.0,
        "min_seconds": float(np.min(dur_array)) if dur_array.size else 0.0,
        "max_seconds": float(np.max(dur_array)) if dur_array.size else 0.0,
        "mean_seconds": float(np.mean(dur_array)) if dur_array.size else 0.0,
        "median_seconds": float(np.median(dur_array)) if dur_array.size else 0.0,
        "std_seconds": float(np.std(dur_array)) if dur_array.size else 0.0,
    }

    genuine_count = sum(1 for r in records if r["class_label"] == "genuine")
    synthetic_count = sum(1 for r in records if r["class_label"] == "synthetic")
    total_valid = len(records)

    summary: Dict[str, Any] = {
        "dataset_root": str(base_data),
        "preprocessed_root": str(out_base),
        "target_sample_rate": target_sr,
        "total_recordings_processed": total_valid,
        "total_genuine_recordings": genuine_count,
        "total_synthetic_recordings": synthetic_count,
        "class_distribution": {
            "genuine": genuine_count,
            "synthetic": synthetic_count,
            "genuine_percentage": float(100.0 * genuine_count / total_valid) if total_valid else 0.0,
            "synthetic_percentage": float(100.0 * synthetic_count / total_valid) if total_valid else 0.0,
        },
        "sample_rate_distribution": {str(k): v for k, v in sorted(sample_rates.items())},
        "audio_duration_statistics": duration_stats,
        "missing_or_corrupted_files": missing_or_corrupted,
        "missing_or_corrupted_count": len(missing_or_corrupted),
        "files": records,
    }

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_dataset_summary(
    dataset_dir: Optional[AudioPath] = None,
    manifest_path: Optional[AudioPath] = None,
) -> Dict[str, Any]:
    """Return the dataset summary, loading from manifest if cached or computing it."""
    from src.utils import project_root

    root = project_root()
    manifest_file = Path(manifest_path) if manifest_path else root / "dataset" / "evidence_manifest.json"
    if manifest_file.is_file():
        with manifest_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    return preprocess_dataset(dataset_dir=dataset_dir, manifest_path=manifest_file)
