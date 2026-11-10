"""SQLite database management and data access layer for forensic cases and evidence."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from database.models import (
    AnalysisFinding,
    AnalysisRun,
    EvidenceComparisonRecord,
    ForensicCase,
    ForensicEvidence,
    ForensicExamination,
    IntegrityRecord,
    User,
    UserRole,
    UserSession,
)
from src.utils import project_root

logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Return default SQLite database filepath in database/forensics.db."""
    root = project_root()
    db_dir = root / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "forensics.db"


def init_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Initialize SQLite database tables and foreign keys."""
    path = Path(db_path) if db_path else get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        # 1. Cases Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                investigator TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE'
            );
        """)

        # 2. Evidence Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT UNIQUE NOT NULL,
                case_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                processed_path TEXT,
                file_size_bytes INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                container_format TEXT NOT NULL,
                sha256_hash TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
            );
        """)

        # 3. Examinations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS examinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                examination_id TEXT UNIQUE NOT NULL,
                evidence_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_synthetic REAL,
                prob_genuine REAL,
                decision_score REAL,
                status TEXT NOT NULL,
                estimated_snr_db REAL,
                clipping_ratio REAL,
                silence_ratio REAL,
                is_in_distribution INTEGER NOT NULL DEFAULT 1,
                max_z_score REAL,
                multi_model_consensus TEXT,
                forensic_assessment TEXT NOT NULL,
                overview_figure_path TEXT,
                report_md_path TEXT,
                report_pdf_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
            );
        """)

        # 4. Integrity Checks Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrity_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT NOT NULL,
                recorded_sha256 TEXT NOT NULL,
                checked_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                verified_by TEXT NOT NULL,
                notes TEXT,
                checked_at TEXT NOT NULL,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
            );
        """)

        # 5. Audit Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                evidence_id TEXT,
                case_id TEXT,
                details_json TEXT NOT NULL
            );
        """)

        # 6. Analysis Runs Table (Phase 7 Intelligence)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                case_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                examination_id TEXT,
                analysis_type TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                algorithm_version TEXT NOT NULL,
                software_versions_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                results_json TEXT NOT NULL,
                output_files_json TEXT NOT NULL,
                model_id TEXT,
                model_version TEXT,
                feature_schema_version TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
            );
        """)

        # 7. Analysis Findings Table (Phase 7 Signal-Level Timestamped Indicators)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id TEXT UNIQUE NOT NULL,
                run_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                indicator TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL,
                interpretation_flag TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
            );
        """)

        # 8. Evidence Comparisons Table (Phase 7 Multi-Evidence Correlation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_id TEXT UNIQUE NOT NULL,
                case_id TEXT NOT NULL,
                evidence_a_id TEXT NOT NULL,
                evidence_b_id TEXT NOT NULL,
                sha_equal INTEGER NOT NULL,
                metadata_diff_json TEXT NOT NULL,
                feature_distance_json TEXT NOT NULL,
                referenced_run_ids_json TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
                FOREIGN KEY (evidence_a_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                FOREIGN KEY (evidence_b_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
            );
        """)

        # 9. Users Table (Phase 8 Authentication & RBAC)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt_hex TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );
        """)

        # 10. Sessions Table (Phase 8 Session Management)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                is_revoked INTEGER NOT NULL DEFAULT 0,
                ip_address TEXT NOT NULL DEFAULT '127.0.0.1',
                user_agent TEXT NOT NULL DEFAULT 'ForensicDashboard/1.0',
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        # 11. Schema Version Tracking (Phase 8 Reliability)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );
        """)

        # Record initial schema version
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version, applied_at, description)
            VALUES ('8.0.0', datetime('now'), 'Phase 8 Production Hardening & RBAC Schema');
        """)

        conn.commit()

        # Enable Write-Ahead Logging (WAL) and performance tuning outside active transaction
        old_iso = conn.isolation_level
        conn.isolation_level = None
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.isolation_level = old_iso

    logger.info("Initialized forensic SQLite database with WAL mode at %s", path)
    return path


class ForensicDatabase:
    """Thread-safe SQLite database manager for forensic case management."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db_path = init_db(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        return conn

    # =========================================================================
    # Case Operations
    # =========================================================================
    def create_case(self, case: ForensicCase) -> ForensicCase:
        """Insert a new forensic case."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, title, description, investigator, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (case.case_id, case.title, case.description, case.investigator, case.created_at, case.status),
            )
            conn.commit()
        return case

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a case record by case_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_cases(self, search: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List forensic cases with optional search and status filtering."""
        query = "SELECT * FROM cases WHERE 1=1"
        params: List[Any] = []

        if status and status != "ALL":
            query += " AND status = ?"
            params.append(status)

        if search:
            query += " AND (case_id LIKE ? OR title LIKE ? OR investigator LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term, term])

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def update_case_status(self, case_id: str, new_status: str) -> bool:
        """Update lifecycle status of a case."""
        with self._get_connection() as conn:
            cursor = conn.execute("UPDATE cases SET status = ? WHERE case_id = ?", (new_status, case_id))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Evidence Operations
    # =========================================================================
    def add_evidence(self, evidence: ForensicEvidence) -> ForensicEvidence:
        """Register a new digital evidence exhibit under a case."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evidence (
                    evidence_id, case_id, original_filename, stored_path, processed_path,
                    file_size_bytes, duration_seconds, sample_rate, channels,
                    container_format, sha256_hash, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id, evidence.case_id, evidence.original_filename,
                    evidence.stored_path, evidence.processed_path, evidence.file_size_bytes,
                    evidence.duration_seconds, evidence.sample_rate, evidence.channels,
                    evidence.container_format, evidence.sha256_hash, evidence.uploaded_at
                ),
            )
            conn.commit()
        return evidence

    def get_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence exhibit by evidence_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_evidence_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        """List all evidence exhibits registered under a case."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM evidence WHERE case_id = ? ORDER BY uploaded_at DESC", (case_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def list_all_evidence(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all evidence exhibits across all cases."""
        query = "SELECT * FROM evidence WHERE 1=1"
        params: List[Any] = []
        if search:
            query += " AND (evidence_id LIKE ? OR original_filename LIKE ? OR sha256_hash LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term, term])
        query += " ORDER BY uploaded_at DESC"
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    # =========================================================================
    # Examination Operations
    # =========================================================================
    def add_examination(self, exam: ForensicExamination) -> ForensicExamination:
        """Record forensic examination results."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO examinations (
                    examination_id, evidence_id, case_id, model_name, predicted_label,
                    confidence, prob_synthetic, prob_genuine, decision_score, status,
                    estimated_snr_db, clipping_ratio, silence_ratio, is_in_distribution,
                    max_z_score, multi_model_consensus, forensic_assessment,
                    overview_figure_path, report_md_path, report_pdf_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exam.examination_id, exam.evidence_id, exam.case_id, exam.model_name,
                    exam.predicted_label, exam.confidence, exam.prob_synthetic,
                    exam.prob_genuine, exam.decision_score, exam.status, exam.estimated_snr_db,
                    exam.clipping_ratio, exam.silence_ratio, 1 if exam.is_in_distribution else 0,
                    exam.max_z_score, exam.multi_model_consensus, exam.forensic_assessment,
                    exam.overview_figure_path, exam.report_md_path, exam.report_pdf_path,
                    exam.created_at
                ),
            )
            conn.commit()
        return exam

    def list_examinations(
        self,
        case_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query examination history with flexible filters."""
        query = "SELECT * FROM examinations WHERE 1=1"
        params: List[Any] = []

        if case_id:
            query += " AND case_id = ?"
            params.append(case_id)
        if evidence_id:
            query += " AND evidence_id = ?"
            params.append(evidence_id)
        if label and label != "ALL":
            query += " AND predicted_label = ?"
            params.append(label)

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_latest_examination_for_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent examination for a given evidence ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM examinations WHERE evidence_id = ? ORDER BY created_at DESC LIMIT 1",
                (evidence_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Integrity Check Operations
    # =========================================================================
    def record_integrity_check(self, check: IntegrityRecord) -> IntegrityRecord:
        """Log an on-demand cryptographic integrity verification event."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO integrity_checks (
                    evidence_id, recorded_sha256, checked_sha256, status,
                    verified_by, notes, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    check.evidence_id, check.recorded_sha256, check.checked_sha256,
                    check.status, check.verified_by, check.notes, check.checked_at
                ),
            )
            conn.commit()
        return check

    def list_integrity_checks(self, evidence_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List integrity check history."""
        query = "SELECT * FROM integrity_checks WHERE 1=1"
        params: List[Any] = []
        if evidence_id:
            query += " AND evidence_id = ?"
            params.append(evidence_id)
        query += " ORDER BY checked_at DESC"
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    # =========================================================================
    # Audit Events Operations
    # =========================================================================
    def log_audit_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        evidence_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> None:
        """Append an auditable event to the database audit_events table."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (timestamp_utc, event_type, evidence_id, case_id, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, event_type, evidence_id, case_id, json.dumps(details)),
            )
            conn.commit()

    def list_audit_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent audit events."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_events ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    # =========================================================================
    # Metrics & KPI Analytics for Dashboard Home
    # =========================================================================
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Aggregate high-level KPI counts for the dashboard home screen."""
        with self._get_connection() as conn:
            c_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            c_evidence = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            c_exams = conn.execute("SELECT COUNT(*) FROM examinations").fetchone()[0]
            c_real = conn.execute("SELECT COUNT(*) FROM examinations WHERE predicted_label = 'GENUINE'").fetchone()[0]
            c_fake = conn.execute("SELECT COUNT(*) FROM examinations WHERE predicted_label = 'SYNTHETIC'").fetchone()[0]
            c_indet = conn.execute("SELECT COUNT(*) FROM examinations WHERE predicted_label = 'INDETERMINATE'").fetchone()[0]

            return {
                "total_cases": c_cases,
                "total_evidence": c_evidence,
                "total_examinations": c_exams,
                "genuine_count": c_real,
                "synthetic_count": c_fake,
                "indeterminate_count": c_indet,
            }

    # =========================================================================
    # Phase 7 Analysis Runs Operations
    # =========================================================================
    def create_analysis_run(self, run: AnalysisRun) -> AnalysisRun:
        """Insert a Phase 7 forensic analysis run record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, case_id, evidence_id, examination_id, analysis_type,
                    algorithm_name, algorithm_version, software_versions_json, parameters_json,
                    input_sha256, status, warnings_json, results_json, output_files_json,
                    model_id, model_version, feature_schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.case_id,
                    run.evidence_id,
                    run.examination_id,
                    run.analysis_type,
                    run.algorithm_name,
                    run.algorithm_version,
                    run.software_versions_json,
                    run.parameters_json,
                    run.input_sha256,
                    run.status,
                    run.warnings_json,
                    run.results_json,
                    run.output_files_json,
                    run.model_id,
                    run.model_version,
                    run.feature_schema_version,
                    run.created_at,
                ),
            )
            conn.commit()
        return run

    def get_analysis_run(self, run_id: str) -> Optional[AnalysisRun]:
        """Retrieve a specific analysis run by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data.pop("id", None)
            return AnalysisRun(**data)

    def list_analysis_runs(
        self,
        case_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> List[AnalysisRun]:
        """List analysis runs filtered by case, evidence, or type."""
        query = "SELECT * FROM analysis_runs WHERE 1=1"
        params: List[Any] = []
        if case_id:
            query += " AND case_id = ?"
            params.append(case_id)
        if evidence_id:
            query += " AND evidence_id = ?"
            params.append(evidence_id)
        if analysis_type:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            runs = []
            for r in cursor.fetchall():
                d = dict(r)
                d.pop("id", None)
                runs.append(AnalysisRun(**d))
            return runs

    # =========================================================================
    # Phase 7 Analysis Findings Operations
    # =========================================================================
    def create_analysis_finding(self, finding: AnalysisFinding) -> AnalysisFinding:
        """Insert a timestamped forensic finding."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_findings (
                    finding_id, run_id, evidence_id, start_sec, end_sec, indicator,
                    value, unit, threshold, severity, interpretation_flag, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.finding_id,
                    finding.run_id,
                    finding.evidence_id,
                    finding.start_sec,
                    finding.end_sec,
                    finding.indicator,
                    finding.value,
                    finding.unit,
                    finding.threshold,
                    finding.severity,
                    finding.interpretation_flag,
                    finding.description,
                    finding.created_at,
                ),
            )
            conn.commit()
        return finding

    def create_analysis_findings_batch(self, findings: List[AnalysisFinding]) -> None:
        """Insert multiple forensic findings in a single transaction."""
        if not findings:
            return
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO analysis_findings (
                    finding_id, run_id, evidence_id, start_sec, end_sec, indicator,
                    value, unit, threshold, severity, interpretation_flag, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f.finding_id,
                        f.run_id,
                        f.evidence_id,
                        f.start_sec,
                        f.end_sec,
                        f.indicator,
                        f.value,
                        f.unit,
                        f.threshold,
                        f.severity,
                        f.interpretation_flag,
                        f.description,
                        f.created_at,
                    )
                    for f in findings
                ],
            )
            conn.commit()

    def list_analysis_findings(
        self,
        run_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> List[AnalysisFinding]:
        """List analysis findings filtered by run or evidence ID."""
        query = "SELECT * FROM analysis_findings WHERE 1=1"
        params: List[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if evidence_id:
            query += " AND evidence_id = ?"
            params.append(evidence_id)
        query += " ORDER BY start_sec ASC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            findings = []
            for r in cursor.fetchall():
                d = dict(r)
                d.pop("id", None)
                findings.append(AnalysisFinding(**d))
            return findings

    # =========================================================================
    # Phase 7 Evidence Comparison Operations
    # =========================================================================
    def create_evidence_comparison(self, comp: EvidenceComparisonRecord) -> EvidenceComparisonRecord:
        """Insert a multi-evidence correlation record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evidence_comparisons (
                    comparison_id, case_id, evidence_a_id, evidence_b_id,
                    sha_equal, metadata_diff_json, feature_distance_json,
                    referenced_run_ids_json, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comp.comparison_id,
                    comp.case_id,
                    comp.evidence_a_id,
                    comp.evidence_b_id,
                    1 if comp.sha_equal else 0,
                    comp.metadata_diff_json,
                    comp.feature_distance_json,
                    comp.referenced_run_ids_json,
                    comp.notes,
                    comp.created_at,
                ),
            )
            conn.commit()
        return comp

    def list_evidence_comparisons(self, case_id: Optional[str] = None) -> List[EvidenceComparisonRecord]:
        """List evidence comparisons filtered by case ID."""
        query = "SELECT * FROM evidence_comparisons WHERE 1=1"
        params: List[Any] = []
        if case_id:
            query += " AND case_id = ?"
            params.append(case_id)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            comps = []
            for r in cursor.fetchall():
                d = dict(r)
                d.pop("id", None)
                d["sha_equal"] = bool(d["sha_equal"])
                comps.append(EvidenceComparisonRecord(**d))
            return comps

    def get_evidence_comparison(self, comparison_id: str) -> Optional[EvidenceComparisonRecord]:
        """Retrieve a specific evidence comparison by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM evidence_comparisons WHERE comparison_id = ?", (comparison_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d.pop("id", None)
            d["sha_equal"] = bool(d["sha_equal"])
            return EvidenceComparisonRecord(**d)

    # =========================================================================
    # Phase 8 User Operations (Authentication & RBAC)
    # =========================================================================
    def create_user(self, user: User) -> User:
        """Insert a new user record with hashed password."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id, username, password_hash, salt_hex, role, full_name,
                    email, is_active, failed_attempts, locked_until, created_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username.strip().lower(),
                    user.password_hash,
                    user.salt_hex,
                    user.role,
                    user.full_name,
                    user.email,
                    1 if user.is_active else 0,
                    user.failed_attempts,
                    user.locked_until,
                    user.created_at,
                    user.last_login_at,
                ),
            )
            conn.commit()
        return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Fetch user record by normalized username."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d.pop("id", None)
            d["is_active"] = bool(d["is_active"])
            return User(**d)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetch user record by unique user_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d.pop("id", None)
            d["is_active"] = bool(d["is_active"])
            return User(**d)

    def list_users(self) -> List[User]:
        """List all registered forensic users."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users ORDER BY created_at ASC")
            users = []
            for r in cursor.fetchall():
                d = dict(r)
                d.pop("id", None)
                d["is_active"] = bool(d["is_active"])
                users.append(User(**d))
            return users

    def update_user(self, user: User) -> None:
        """Update user record attributes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE users SET
                    password_hash = ?, salt_hex = ?, role = ?, full_name = ?,
                    email = ?, is_active = ?, failed_attempts = ?, locked_until = ?, last_login_at = ?
                WHERE user_id = ?
                """,
                (
                    user.password_hash,
                    user.salt_hex,
                    user.role,
                    user.full_name,
                    user.email,
                    1 if user.is_active else 0,
                    user.failed_attempts,
                    user.locked_until,
                    user.last_login_at,
                    user.user_id,
                ),
            )
            conn.commit()

    def record_login_success(self, user_id: str) -> None:
        """Reset failed attempts and record login timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = ? WHERE user_id = ?",
                (now, user_id),
            )
            conn.commit()

    def record_login_failure(self, user_id: str, lock_threshold: int = 5, lock_minutes: int = 15) -> Tuple[int, Optional[str]]:
        """Increment failed attempts and lock account if threshold exceeded."""
        now = datetime.now(timezone.utc)
        with self._get_connection() as conn:
            row = conn.execute("SELECT failed_attempts FROM users WHERE user_id = ?", (user_id,)).fetchone()
            attempts = (row[0] if row else 0) + 1
            locked_until_str = None

            if attempts >= lock_threshold:
                locked_dt = now + timedelta(minutes=lock_minutes)
                locked_until_str = locked_dt.isoformat()

            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE user_id = ?",
                (attempts, locked_until_str, user_id),
            )
            conn.commit()
            return attempts, locked_until_str

    # =========================================================================
    # Phase 8 Session Operations
    # =========================================================================
    def create_session(self, session: UserSession) -> UserSession:
        """Insert a new active session record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, username, role, created_at, expires_at,
                    is_revoked, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.user_id,
                    session.username,
                    session.role,
                    session.created_at,
                    session.expires_at,
                    1 if session.is_revoked else 0,
                    session.ip_address,
                    session.user_agent,
                ),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Fetch and validate a session record."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d.pop("id", None)
            d["is_revoked"] = bool(d["is_revoked"])
            return UserSession(**d)

    def revoke_session(self, session_id: str) -> None:
        """Mark a session as revoked."""
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET is_revoked = 1 WHERE session_id = ?", (session_id,))
            conn.commit()

    def cleanup_expired_sessions(self) -> int:
        """Delete sessions that have passed their expiration timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at < ? OR is_revoked = 1", (now,))
            deleted = cursor.rowcount
            conn.commit()
            return deleted



