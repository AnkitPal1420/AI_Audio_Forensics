# Phase 8: Forensic Platform Administrator Operating Guide

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 8.12 — Administrator Guide  
**Audience:** System Administrators, Forensic Lab Directors, IT Security Officers  
**Date:** November 2026  

---

## 1. System Deployment & Startup

### 1.1 First-Time Installation
To deploy the application on a new forensic workstation:
1. Open Windows PowerShell as an Administrator.
2. Navigate to the project root:
   ```powershell
   cd c:\Users\ANKIT\OneDrive\Desktop\AI_Audio_Forensics
   ```
3. Run the automated deployment script:
   ```powershell
   .\scripts\deploy.ps1
   ```
   *The script creates required directory hierarchies, sets up the SQLite database with WAL concurrency, verifies all model artifact hashes, and prompts for initial Administrator account setup.*

### 1.2 Starting the Hardened Dashboard
Launch the dashboard strictly bound to local loopback:
```powershell
.\scripts\start.ps1
```
The application will be accessible locally at `http://127.0.0.1:8501`. Network binding to `0.0.0.0` is prevented.

---

## 2. User & Privilege Management

### 2.1 Role Definitions
- **Administrator:** Full system control. Can provision users, trigger hot backups, configure models, and view system health.
- **Forensic Examiner:** Day-to-day casework. Can create cases, upload audio evidence, run AI detection, compute XAI attributions, and generate reports.
- **Reviewing Officer:** Peer review and supervisory oversight. Can review completed case files, inspect acoustic findings, approve reports, and add review comments. Cannot run new ML analyses or alter cases.
- **Read-only User:** External investigators or legal counsel. Can view case records and download authorized PDF reports.

### 2.2 Provisioning a New User
1. Log into the dashboard with an **Administrator** account.
2. Navigate to **`⚙️ System Settings & Diagnostics`** $\to$ **`👥 Forensic User Management`**.
3. Under **`➕ Provision New Forensic User`**, supply:
   - Unique Username (case-insensitive)
   - Full Legal Name
   - Assigned Role (`Forensic Examiner`, `Reviewing Officer`, `Read-only User`)
   - Temporary Password (must satisfy complexity: $\ge 10$ characters, upper/lower/digit/symbol).
4. Click **`Register Forensic User`**.

### 2.3 Managing Locked Accounts
- Accounts automatically lock for **15 minutes** after 5 consecutive failed password attempts.
- To immediately unlock an account prior to timeout:
  ```powershell
  .venv314\Scripts\python.exe -c "
  from database.database import ForensicDatabase
  db = ForensicDatabase()
  u = db.get_user_by_username('target_username')
  if u:
      u.failed_attempts = 0
      u.locked_until = None
      db.update_user(u)
      print('User unlocked successfully.')
  "
  ```

---

## 3. Hot Backup & Disaster Recovery Operations

### 3.1 Triggering an On-Demand Hot Backup
1. In the dashboard, navigate to **`⚙️ System Settings & Diagnostics`** $\to$ **`💾 Hot Online Backups`**.
2. Click **`🚀 Trigger Online Hot System Backup`**.
3. Alternatively, execute via PowerShell:
   ```powershell
   .\scripts\backup.ps1
   ```
   *Backups are non-blocking and execute safely while examiners are actively using the system.*

### 3.2 Restoring from a Backup
To restore the platform from an archive snapshot:
```powershell
.venv314\Scripts\python.exe -c "
from src.backup_manager import restore_system_backup
res = restore_system_backup('database/backups/forensic_backup_YYYYMMDD_HHMMSS.zip')
print('Restoration status:', res['status'])
"
```
The restoration engine automatically verifies the archive's SHA-256 seal and validates database integrity via `PRAGMA integrity_check;`.

---

## 4. Log Inspection & Auditing

- **Operational Logs:** [`logs/operational.log`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/logs/operational.log) — Audio decoding steps, model loads, warning events.
- **Security Logs:** [`logs/security.log`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/logs/security.log) — Authentication successes, failed logins, lockout triggers.
- **Evidentiary Audit Stream:** [`logs/audit_events.jsonl`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/logs/audit_events.jsonl) — Immutable exhibit intake records, hash verifications, and report generations.

