"""Authentication, Password Security & Role-Based Access Control (RBAC) Module.

Adheres to NIST SP 800-63B and OWASP Application Security Verification Standards:
- Memory-hard password hashing via hashlib.scrypt (N=16384, r=8, p=1).
- 256-bit cryptographic salt generation via secrets.token_bytes(32).
- Constant-time verification using hmac.compare_digest to prevent timing side channels.
- Brute-force protection: 5-attempt threshold with automatic 15-minute account lock.
- Cryptographically secure 256-bit session tokens with expiration and revocation.
- Strict 4-tier Role-Based Access Control (RBAC) matrix.
"""

import hashlib
import hmac
import logging
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.database import ForensicDatabase
from database.models import User, UserRole, UserSession

logger = logging.getLogger(__name__)

# Scrypt KDF Security Parameters (OWASP recommended for interactive logins)
SCRYPT_N = 16384  # CPU/memory cost
SCRYPT_R = 8      # Block size
SCRYPT_P = 1      # Parallelization parameter
SCRYPT_DKLEN = 64 # Derived key length in bytes
SCRYPT_MAXMEM = 64 * 1024 * 1024 # 64 MB RAM limit

LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
DEFAULT_SESSION_HOURS = 8


class AuthenticationError(RuntimeError):
    """Raised on invalid credentials or security rejection."""
    pass


class AccountLockedError(AuthenticationError):
    """Raised when an account is temporarily locked due to excessive failed attempts."""
    pass


class InsufficientPermissionsError(PermissionError):
    """Raised when an authenticated user lacks authorization for an action."""
    pass


# =============================================================================
# Password Hashing & Policy Validation
# =============================================================================
def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Derive scrypt hash and return (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_bytes(32)

    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=SCRYPT_DKLEN,
    )
    return derived.hex(), salt.hex()


def verify_password(password: str, stored_hash_hex: str, salt_hex: str) -> bool:
    """Verify password in constant time using hmac.compare_digest."""
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(stored_hash_hex)
    except ValueError:
        return False

    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=SCRYPT_DKLEN,
    )
    return hmac.compare_digest(derived, expected)


def validate_password_complexity(password: str) -> Tuple[bool, str]:
    """Enforce NIST-aligned forensic password complexity standards."""
    if len(password) < 10:
        return False, "Password must be at least 10 characters in length."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one numerical digit (0-9)."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]]', password):
        return False, "Password must contain at least one special character."
    return True, "Password meets forensic complexity criteria."


# =============================================================================
# User Authentication & Lockout Handling
# =============================================================================
def authenticate_user(
    username: str,
    password: str,
    db: ForensicDatabase,
    ip_address: str = "127.0.0.1",
) -> User:
    """Authenticate a user against the database with brute-force lockout checking."""
    user = db.get_user_by_username(username)
    now = datetime.now(timezone.utc)

    if not user:
        # Constant-time dummy computation to mitigate timing enumeration
        dummy_salt = secrets.token_bytes(32)
        hash_password(password, dummy_salt)
        logger.warning("Failed login attempt for non-existent user '%s' from IP %s", username, ip_address)
        raise AuthenticationError("Invalid username or password.")

    if not user.is_active:
        logger.warning("Login rejected for deactivated user '%s' from IP %s", username, ip_address)
        raise AuthenticationError("This forensic investigator account is deactivated.")

    # Check Lockout status
    if user.locked_until:
        try:
            locked_dt = datetime.fromisoformat(user.locked_until)
            if now < locked_dt:
                remaining_min = int((locked_dt - now).total_seconds() // 60) + 1
                logger.warning("Login rejected for locked account '%s' (locked for %d more mins)", username, remaining_min)
                raise AccountLockedError(f"Account is locked due to security policy. Please try again in {remaining_min} minute(s).")
        except ValueError:
            pass

    # Verify Password
    is_valid = verify_password(password, user.password_hash, user.salt_hex)

    if not is_valid:
        attempts, locked_until = db.record_login_failure(
            user.user_id, lock_threshold=LOCKOUT_ATTEMPTS, lock_minutes=LOCKOUT_MINUTES
        )
        logger.warning("Failed password attempt #%d for user '%s' from IP %s", attempts, username, ip_address)
        if locked_until:
            raise AccountLockedError(f"Account locked for {LOCKOUT_MINUTES} minutes due to {LOCKOUT_ATTEMPTS} consecutive failed attempts.")
        raise AuthenticationError(f"Invalid username or password. ({LOCKOUT_ATTEMPTS - attempts} attempt(s) remaining)")

    # Login Succeeded
    db.record_login_success(user.user_id)
    logger.info("User '%s' [%s] successfully authenticated from IP %s", user.username, user.role, ip_address)
    return user


# =============================================================================
# Session Lifecycle Management
# =============================================================================
def create_user_session(
    user: User,
    db: ForensicDatabase,
    duration_hours: int = DEFAULT_SESSION_HOURS,
    ip_address: str = "127.0.0.1",
    user_agent: str = "ForensicDashboard/1.0",
) -> UserSession:
    """Generate cryptographically secure session token and persist in sessions table."""
    session_id = secrets.token_hex(32)  # 256-bit secure token
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=duration_hours)

    session = UserSession(
        session_id=session_id,
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        created_at=now.isoformat(),
        expires_at=expires.isoformat(),
        is_revoked=False,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.create_session(session)
    return session


def validate_session(session_id: Optional[str], db: ForensicDatabase) -> Optional[UserSession]:
    """Validate a session token: verify existence, non-revocation, and non-expiration."""
    if not session_id:
        return None

    session = db.get_session(session_id)
    if not session or session.is_revoked:
        return None

    now = datetime.now(timezone.utc)
    try:
        exp_dt = datetime.fromisoformat(session.expires_at)
        if now >= exp_dt:
            db.revoke_session(session_id)
            return None
    except ValueError:
        return None

    return session


# =============================================================================
# Role-Based Access Control (RBAC) Matrix
# =============================================================================
RBAC_PERMISSIONS: Dict[str, List[str]] = {
    UserRole.ADMINISTRATOR: [
        "view_cases",
        "create_cases",
        "delete_cases",
        "upload_evidence",
        "run_examinations",
        "run_xai",
        "run_tampering_analysis",
        "correlate_evidence",
        "export_reports",
        "manage_users",
        "view_audit_trail",
        "manage_backups",
        "view_system_diagnostics",
    ],
    UserRole.FORENSIC_EXAMINER: [
        "view_cases",
        "create_cases",
        "upload_evidence",
        "run_examinations",
        "run_xai",
        "run_tampering_analysis",
        "correlate_evidence",
        "export_reports",
        "view_audit_trail",
    ],
    UserRole.REVIEWING_OFFICER: [
        "view_cases",
        "view_examinations",
        "export_reports",
        "view_audit_trail",
        "add_review_notes",
    ],
    UserRole.READ_ONLY_USER: [
        "view_cases",
        "view_examinations",
        "export_reports",
    ],
}


def has_permission(role: str, action: str) -> bool:
    """Check if a specific role possesses permission for a designated action."""
    allowed_actions = RBAC_PERMISSIONS.get(role, [])
    return action in allowed_actions


def check_permission(session: Optional[UserSession], action: str) -> bool:
    """Enforce authorization against active session."""
    if not session:
        return False
    return has_permission(session.role, action)


def require_permission(session_id: Optional[str], action: str, db: ForensicDatabase) -> UserSession:
    """Server-side authorization gate: re-validate the session against the database and
    confirm the role currently stored for the user permits `action`.

    The role is re-read from the users table (not from client/session-state copies), so a
    demoted or deactivated account loses access immediately. Raises on any failure.
    """
    session = validate_session(session_id, db)
    if not session:
        raise InsufficientPermissionsError("Authentication required: session missing, expired or revoked.")
    user = db.get_user_by_id(session.user_id)
    if not user or not user.is_active:
        raise InsufficientPermissionsError("Account is deactivated or no longer exists.")
    if not has_permission(user.role, action):
        logger.warning("RBAC denial: user '%s' (%s) attempted '%s'", user.username, user.role, action)
        raise InsufficientPermissionsError(f"Role '{user.role}' is not authorized for '{action}'.")
    return session


# =============================================================================
# Initial User Provisioning (Zero-Hardcoded Secrets)
# =============================================================================
def bootstrap_admin_user(
    db: ForensicDatabase,
    username: str = "admin",
    password: Optional[str] = None,
    full_name: str = "Primary Forensic Administrator",
) -> Tuple[User, str]:
    """Bootstrap initial administrator account if no users exist.
    
    If password is not provided, a random 16-character high-entropy password is generated.
    Returns (user, raw_password).
    """
    existing = db.list_users()
    if existing:
        raise RuntimeError("Cannot bootstrap: user accounts already exist in database.")

    if not password:
        password = secrets.token_urlsafe(14) + "A1!"

    is_valid, msg = validate_password_complexity(password)
    if not is_valid:
        raise ValueError(f"Admin bootstrap password rejected: {msg}")

    pwd_hash, salt_hex = hash_password(password)
    user = User(
        user_id=f"USER-ADMIN-{uuid.uuid4().hex[:8].upper()}",
        username=username.strip().lower(),
        password_hash=pwd_hash,
        salt_hex=salt_hex,
        role=UserRole.ADMINISTRATOR,
        full_name=full_name,
        email="admin@forensics.local",
    )
    db.create_user(user)
    logger.info("Bootstrapped initial Administrator account '%s'", username)
    return user, password

