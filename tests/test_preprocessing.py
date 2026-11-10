"""Tests for audio preprocessing and evidence verification helpers."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio_preprocessing import (
    CorruptedAudioError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    compute_file_hash,
    convert_to_mono,
    get_audio_metadata,
    load_audio,
    normalize_audio,
    preprocess_and_save,
    resample_waveform,
    standardize_audio,
)


def test_normalize_audio_scales_peak_to_one():
    waveform = np.array([-0.25, 0.5, 1.0], dtype=np.float32)

    normalized = normalize_audio(waveform)

    assert np.max(np.abs(normalized)) == pytest.approx(1.0)
    assert normalized.tolist() == pytest.approx([-0.25, 0.5, 1.0])


def test_normalize_audio_preserves_silence():
    waveform = np.zeros(8, dtype=np.float32)

    normalized = normalize_audio(waveform)

    assert np.array_equal(normalized, waveform)
    assert normalized is not waveform


def test_normalize_audio_rejects_non_mono_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_audio(np.zeros((2, 8), dtype=np.float32))


def test_convert_to_mono_averages_stereo():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0], [0.5, 1.5]], dtype=np.float32)
    mono = convert_to_mono(stereo)
    assert mono.ndim == 1
    assert mono.shape == (3,)
    assert mono.tolist() == pytest.approx([2.0, 3.0, 1.0])


def test_resample_waveform_changes_sample_count():
    orig_sr = 32_000
    target_sr = 16_000
    t = np.arange(orig_sr // 2, dtype=np.float32) / orig_sr
    wave = np.sin(2 * np.pi * 440 * t)

    resampled = resample_waveform(wave, orig_sr, target_sr)
    assert resampled.size == 8000
    assert resampled.dtype == np.float32


def test_compute_file_hash_deterministic(tmp_path: Path):
    sample_file = tmp_path / "evidence.bin"
    sample_file.write_bytes(b"FORENSIC_AUDIO_EVIDENCE_DATA_12345")

    hash1 = compute_file_hash(sample_file)
    hash2 = compute_file_hash(sample_file)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string


def test_compute_file_hash_missing_file():
    with pytest.raises(FileNotFoundError):
        compute_file_hash("nonexistent_path_to_audio_evidence.wav")


def test_get_audio_metadata_valid_wav(tmp_path: Path):
    audio_path = tmp_path / "valid.wav"
    sr = 22_050
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    sf.write(str(audio_path), samples, sr)

    meta = get_audio_metadata(audio_path)
    assert meta["status"] == "valid"
    assert meta["sample_rate"] == 22_050
    assert meta["channels"] == 1
    assert meta["duration_seconds"] == pytest.approx(1.0, rel=1e-2)
    assert "sha256" in meta


def test_get_audio_metadata_empty_file(tmp_path: Path):
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")

    meta = get_audio_metadata(empty_path)
    assert meta["status"] == "empty"
    assert meta["file_size_bytes"] == 0


def test_get_audio_metadata_corrupted_file(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt invalid audio header bytes")

    meta = get_audio_metadata(corrupt_path)
    assert meta["status"] == "corrupted"
    assert meta["error"] is not None


def test_get_audio_metadata_unsupported_format(tmp_path: Path):
    bad_path = tmp_path / "test.txt"
    bad_path.write_text("Not audio", encoding="utf-8")

    meta = get_audio_metadata(bad_path)
    assert meta["status"] == "unsupported"


def test_load_audio_resampling_and_mono(tmp_path: Path):
    audio_path = tmp_path / "stereo_48k.wav"
    sr_orig = 48_000
    t = np.arange(sr_orig // 4, dtype=np.float32) / sr_orig
    stereo_wave = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * t), 0.4 * np.sin(2 * np.pi * 880 * t)])
    sf.write(str(audio_path), stereo_wave, sr_orig)

    loaded, target_sr = load_audio(audio_path, sample_rate=16_000, mono=True)
    assert target_sr == 16_000
    assert loaded.ndim == 1
    assert loaded.size == 4000
    assert loaded.dtype == np.float32


def test_load_audio_rejects_empty_file(tmp_path: Path):
    empty_file = tmp_path / "zero.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(EmptyAudioError):
        load_audio(empty_file)


def test_load_audio_rejects_corrupted_file(tmp_path: Path):
    corrupt_file = tmp_path / "damaged.wav"
    corrupt_file.write_bytes(b"NOT_A_VALID_WAV_HEADER_DATA")
    with pytest.raises(CorruptedAudioError):
        load_audio(corrupt_file)


def test_load_audio_rejects_unsupported_extension(tmp_path: Path):
    bad_ext = tmp_path / "audio.xyz"
    bad_ext.write_bytes(b"some binary payload")
    with pytest.raises(UnsupportedAudioFormatError):
        load_audio(bad_ext)


def test_preprocess_and_save_preserves_original(tmp_path: Path):
    orig_path = tmp_path / "evidence_orig.wav"
    sr = 44_100
    t = np.arange(sr // 2, dtype=np.float32) / sr
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(orig_path), wave, sr)

    orig_hash_before = compute_file_hash(orig_path)
    derivative_path = tmp_path / "derivatives" / "derivative_16k.wav"

    info = preprocess_and_save(orig_path, derivative_path, target_sr=16_000, normalize=True)

    # Verify original file was completely preserved
    orig_hash_after = compute_file_hash(orig_path)
    assert orig_hash_before == orig_hash_after
    assert info["source_sha256"] == orig_hash_before
    assert info["derivative_sample_rate"] == 16_000
    assert info["derivative_channels"] == 1
    assert Path(derivative_path).is_file()
