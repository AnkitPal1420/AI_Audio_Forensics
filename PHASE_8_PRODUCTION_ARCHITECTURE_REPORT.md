# Phase 8.1: Production Architecture & Deployment Environment Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Operating System:** Windows 11  
**Python Environment:** `.venv314` (Python 3.14.3)  
**Database:** SQLite (`database/forensics.db`)  
**Web Framework:** Streamlit 1.67.0  
**Phase:** 8.1 — Production Architecture  
**Date:** November 2026  
**Status:** IMPLEMENTED & AUDITED  

---

## 1. Executive Summary

Phase 8.1 defines the production deployment topology, security perimeter, and execution constraints for the AI Audio Forensics platform. Forensic software handling sensitive criminal evidence must never be deployed on untrusted public networks or with default permissive settings.

This architecture report establishes three supported deployment environments, enforces network perimeter restrictions, defines storage permission models, and formalizes operational startup procedures.

---

## 2. Deployment Topology Options

### Deployment Tier 1: Standalone Air-Gapped Forensic Workstation (Primary / Recommended)
- **Use Case:** High-security criminal investigation laboratories, law enforcement evidence vaults.
- **Network Interface:** Local loopback only (`127.0.0.1`). Physical or virtual network adapters disabled or firewall-blocked from outbound internet routing.
- **User Base:** Dedicated forensic examiner operating locally.
- **Storage:** Direct attached encrypted storage (BitLocker / VeraCrypt volume) with NTFS file permissions.
- **Strengths:** Maximum defense against remote network compromise, zero data leakage risk.

### Deployment Tier 2: Private Laboratory Intranet Server (Collaborative Case Review)
- **Use Case:** Regional forensic units where multiple examiners and reviewing officers collaborate.
- **Network Interface:** Bound to private lab VLAN with reverse proxy TLS termination.
- **Access Control:** HTTPS (port 443) fronted by Caddy / Nginx reverse proxy; Streamlit upstream bound to `127.0.0.1:8501`.
- **Strengths:** Centralized evidence storage, peer review workflows, automated backups.

### Deployment Tier 3: Isolated Secure Container (Experimental / Secondary)
- **Use Case:** Reproducible laboratory test harness or staging environments.
- **Constraint:** Windows container with fixed base Python 3.14, volume-mounted evidence directory.

---

## 3. Network Perimeter & Ingress Controls

1. **Default Loopback Enforcement:**  
   The application startup script (`scripts/start.ps1`) enforces:
   ```powershell
   --server.address 127.0.0.1 --server.port 8501 --server.headless true
   ```
   Binding to wildcard (`0.0.0.0`) is strictly disabled in production configuration to prevent exposure to external LAN subnets.

2. **Reverse Proxy Configuration (for Tier 2):**  
   For private laboratory networks, all client communication must terminate at a hardened reverse proxy with modern TLS 1.3 ciphers and the following security headers:
   - `Content-Security-Policy: default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;`
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: strict-origin-when-cross-origin`

---

## 4. Storage & Filesystem Architecture

The filesystem structure strictly segregates untrusted inputs, immutable evidence, derivatives, and models:

```
AI_Audio_Forensics/
 ├── config/                     # Configuration schemas & feature definitions
 ├── database/
 │     ├── forensics.db          # Primary SQLite database (WAL mode enabled)
 │     └── backups/              # Automated hot database backups
 ├── logs/                       # Segregated operational & security logs
 │     ├── operational.log       # System operational events
 │     ├── security.log          # Authentication failures, access denials
 │     └── audit_events.jsonl    # Evidentiary audit trail (ISO/IEC 27037)
 ├── models/                     # Trained ML model artifacts & hash manifest
 │     ├── model_manifest.json   # Cryptographic SHA-256 seal of all .joblib files
 │     └── *.joblib              # Serialized scikit-learn models
 ├── storage/
 │     ├── original_evidence/    # Bitstream-exact original audio (OS Read-Only)
 │     ├── processed_audio/      # Resampled 16 kHz derivatives
 │     ├── visualizations/       # Spectrograms, trajectory charts
 │     ├── reports/              # Court-ready Markdown & PDF reports
 │     └── temp/                 # Ephemeral staging (cleaned after processing)
 └── scripts/                    # Verified PowerShell deployment scripts
```

---

## 5. System Requirements & Resource Limits

| Resource | Minimum Specification | Recommended Production Specification |
| :--- | :--- | :--- |
| **CPU** | 4 Physical Cores (x86_64, AVX2 support) | 8+ Physical Cores (Intel Core i7/i9 or AMD Ryzen 7/9) |
| **RAM** | 8 GB DDR4 | 16 GB - 32 GB DDR4/DDR5 (for long audio STFT & XAI) |
| **Storage** | 20 GB free disk space | 500+ GB High-Speed NVMe SSD (BitLocker Encrypted) |
| **Operating System** | Windows 11 Professional / Enterprise | Windows 11 Enterprise (LTSC / WDAC Enforced) |
| **Python Runtime** | Python 3.14.3 (`.venv314`) | Python 3.14.3 (`.venv314`) pinned dependencies |
| **Max Ingestion File Size** | Configurable: 100 MB default | Configurable: 250 MB ceiling |

---

## 6. Separation of Environments

- **Development (`env = "development"`):** Debug logging enabled, test database seeded, verbose stack traces on error.
- **Production (`env = "production"`):** Strict authentication required, loopback address enforced, generic courtroom-safe error messages surfaced in UI, stack traces routed exclusively to `logs/operational.log`.
