"""Cybersecurity Hardening, Input Validation & Deserialization Defense Module.

Adheres to OWASP Top 10 Security Guidelines:
- Magic-byte validation for audio container formats (preventing polyglot & MIME confusion attacks).
- Maximum payload size enforcement to mitigate Denial of Service (DoS) memory exhaustion.
- Cryptographic SHA-256 verification of serialized model artifacts prior to joblib.load()
  (preventing arbitrary code execution via compromised pickle objects).
- Strict filename and path traversal sanitization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from src.utils import project_root

logger = logging.getLogger(__name__)

# Max upload size ceiling (250 MB default)
DEFAULT_MAX_UPLOAD_BYTES = 250 * 1024 * 1024


class SecurityValidationError(ValueError):
    """Raised when an untrusted input fails security screening."""
    pass


class ModelIntegrityError(RuntimeError):
    """Raised when a serialized machine-learning model artifact fails cryptographic verification."""
    pass


# =============================================================================
# 1. Magic-Byte Audio Container Validation
# =============================================================================
def validate_audio_magic_bytes(data: bytes) -> Tuple[bool, str, str]:
    """Inspect binary header to verify genuine audio container magic bytes.
    
    Returns:
        (is_valid: bool, detected_format: str, message: str)
    """
    if len(data) < 4:
        return False, "UNKNOWN", "Payload too short to verify valid audio header."

    # WAV (RIFF....WAVE)
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return True, "WAV", "Verified RIFF/WAVE container header."

    # FLAC (fLaC)
    if data[:4] == b"fLaC":
        return True, "FLAC", "Verified Free Lossless Audio Codec (fLaC) header."

    # OGG (OggS)
    if data[:4] == b"OggS":
        return True, "OGG", "Verified Ogg Vorbis/Opus container header."

    # MP3 with ID3v2 tag
    if data[:3] == b"ID3":
        return True, "MP3", "Verified MPEG Audio with ID3v2 container tag."

    # Raw MPEG Audio Frame Sync (0xFF followed by MPEG-1/2/2.5 sync bits)
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True, "MP3", "Verified raw MPEG audio frame synchronization header."

    # M4A / MP4 Audio (ftyp box in first 16 bytes)
    if b"ftyp" in data[:16]:
        return True, "M4A", "Verified ISO Base Media File Format (ftyp/M4A) header."

    return False, "UNKNOWN", "Magic bytes do not match any recognized audio container format."


def validate_upload_payload(
    data: bytes,
    filename: str,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> Tuple[bool, str]:
    """Validate uploaded file size, filename characters, and magic bytes."""
    # 1. Size check
    if len(data) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        return False, f"File payload exceeds maximum permitted threshold ({max_mb:.0f} MB)."

    if len(data) == 0:
        return False, "File payload is empty (0 bytes)."

    # 2. Filename traversal check
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, "Filename contains illicit directory traversal tokens."

    # 3. Magic-byte verification
    is_valid, fmt, msg = validate_audio_magic_bytes(data)
    if not is_valid:
        return False, f"Security screening rejected file: {msg}"

    return True, f"Security validation passed ({fmt})."


# =============================================================================
# 2. Cryptographic Model Artifact Verification (Deserialization Defense)
# =============================================================================
def verify_model_artifact(
    model_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Verify that a serialized .joblib model matches its cryptographic SHA-256 manifest.
    
    Prevents execution of malicious pickle payloads via compromised or substituted models.
    """
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}")

    root = project_root()
    if manifest_path is None:
        manifest_path = root / "models" / "model_manifest.json"
    else:
        manifest_path = Path(manifest_path).resolve()

    if not manifest_path.is_file():
        logger.warning("Model manifest not found at %s. Verification cannot proceed.", manifest_path)
        raise ModelIntegrityError(f"Model integrity manifest missing at {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: Dict[str, str] = json.load(f)

    expected_hash = manifest.get(path.name)
    if not expected_hash:
        raise ModelIntegrityError(
            f"Model artifact '{path.name}' is unsealed: missing from official model_manifest.json"
        )

    # Compute actual file digest
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().lower()

    if actual_hash != expected_hash.lower():
        logger.critical(
            "TAMPERING DETECTED: Model %s SHA-256 hash mismatch! Expected %s, got %s",
            path.name,
            expected_hash,
            actual_hash,
        )
        raise ModelIntegrityError(
            f"SECURITY ALERT: Model artifact '{path.name}' failed cryptographic verification! "
            f"File may have been tampered with or corrupted."
        )

    return True
