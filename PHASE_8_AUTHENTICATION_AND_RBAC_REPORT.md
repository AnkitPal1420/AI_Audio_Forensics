# Phase 8.2: Authentication & Role-Based Access Control (RBAC) Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 8.2 — Authentication & Role-Based Access Control  
**Date:** November 2026  
**Status:** IMPLEMENTED & AUDITED  

---

## 1. Executive Summary

Phase 8.2 remediates **SEC-01** (Unauthenticated Dashboard) and **SEC-02** (Absence of RBAC) by establishing a NIST SP 800-63B and OWASP-compliant identity management subsystem (`src/auth.py`). 

The architecture strictly eliminates hardcoded production passwords, enforces memory-hard password hashing, provides automated brute-force account lockouts, generates cryptographically secure 256-bit session tokens, and restricts sensitive case operations according to a 4-tier privilege matrix.

---

## 2. Password Security & Cryptographic Derivation

### 2.1 Memory-Hard Key Derivation (scrypt)
To protect user passwords against offline brute-force, dictionary, and GPU/ASIC rainbow-table attacks, the system utilizes Python 3.14's native `hashlib.scrypt` key derivation function:
- $N = 16,384$ (CPU/memory iteration cost)
- $r = 8$ (block size parameter)
- $p = 1$ (parallelization parameter)
- Salt: 256 bits (32 bytes) of cryptographically secure random bytes generated via `secrets.token_bytes(32)`.
- Derived Key: 512 bits (64 bytes) hex-encoded.

### 2.2 Constant-Time Comparison
Password verification uses `hmac.compare_digest()` to ensure constant-time execution, neutralizing timing side-channel attacks aimed at deriving hash characters.

### 2.3 Password Complexity Standards
All investigator passwords must satisfy:
- Minimum length: $\ge 10$ characters.
- Character diversity: At least one uppercase ($A-Z$), one lowercase ($a-z$), one number ($0-9$), and one special symbol.

---

## 3. Account Lockout & Brute-Force Defense

- Failed login attempts are incremented in the SQLite `users` table (`failed_attempts`).
- Upon reaching **5 consecutive failed attempts**, the account is locked for **15 minutes** (`locked_until = now + 15 min`).
- Subsequent login attempts while locked are immediately rejected with `AccountLockedError`.
- Successful authentication resets `failed_attempts = 0` and clears `locked_until`.
- When an unknown username is submitted, a dummy `scrypt` computation is executed to prevent username enumeration via response-timing discrepancies.

---

## 4. Role-Based Access Control (RBAC) Matrix

The system enforces 4 discrete roles across all application operations:

| Forensic Role | Permitted Actions | Restricted Actions |
| :--- | :--- | :--- |
| **Administrator** | Full system control: user provisioning, case creation/deletion, evidence intake, ML analysis, hot backups, and audit trail inspection. | None. |
| **Forensic Examiner** | Create cases, ingest evidence, run AI predictions, compute XAI attributions, run tampering analysis, export forensic reports. | Cannot delete cases, manage users, or restore backups. |
| **Reviewing Officer** | Inspect active cases, review completed examination records, inspect acoustic findings, export reports, and record review notes. | Cannot upload new evidence, run new ML models, or alter cases. |
| **Read-only User** | View cases, browse completed examinations, and download generated reports. | Strictly read-only access. Cannot execute any mutations or ML pipelines. |

---

## 5. Session Token Security

- Session tokens are 256-bit hexadecimal identifiers generated via `secrets.token_hex(32)`.
- Sessions have an 8-hour default lifespan (`expires_at`).
- Revoked or expired sessions are invalidated on access and purged periodically using `cleanup_expired_sessions()`.
- Session state in the Streamlit web dashboard tracks the authenticated user, role, and active token.

---

## 6. Zero Hardcoded Credentials

The platform provides a secure bootstrapping utility (`bootstrap_admin_user()`). When initializing a fresh laboratory instance, the examiner must either supply a strong initial password or accept an auto-generated 16-character high-entropy secret. No default credentials exist in source code or configuration files.

