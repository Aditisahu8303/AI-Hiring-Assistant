"""
modules/interview_database.py
SQLite persistence layer for interview sessions.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join("database", "interviews.db")


def _connect():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT    NOT NULL,
            email         TEXT,
            role          TEXT,
            experience    TEXT,
            interview_date TEXT,
            total_questions INTEGER,
            technical_score  REAL,
            communication_score REAL,
            relevance_score    REAL,
            completeness_score REAL,
            overall_score      REAL,
            interview_pct      REAL,
            resume_score       REAL,
            recommendation     TEXT,
            status             TEXT,
            top_strengths      TEXT,
            top_improvements   TEXT,
            category_scores    TEXT,
            difficulty_scores  TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS interview_qa (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER REFERENCES interviews(id) ON DELETE CASCADE,
            question_no  INTEGER,
            question     TEXT,
            category     TEXT,
            difficulty   TEXT,
            answer       TEXT,
            technical_score      REAL,
            communication_score  REAL,
            relevance_score      REAL,
            completeness_score   REAL,
            overall_score        REAL,
            strengths   TEXT,
            improvements TEXT,
            feedback    TEXT,
            verdict     TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_interview(
    candidate_name: str,
    email: str,
    role: str,
    experience: str,
    scorecard: dict,
    qa_pairs: list,
) -> int:
    """
    Persist a completed interview.

    Args:
        qa_pairs: [{"question": ..., "category": ..., "difficulty": ...,
                    "answer": ..., <eval fields>}, ...]

    Returns:
        interview_id (int)
    """
    init_db()
    conn = _connect()
    c    = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    c.execute("""
        INSERT INTO interviews (
            candidate_name, email, role, experience, interview_date,
            total_questions, technical_score, communication_score,
            relevance_score, completeness_score, overall_score,
            interview_pct, resume_score, recommendation, status,
            top_strengths, top_improvements, category_scores, difficulty_scores
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        candidate_name, email, role, experience, now,
        scorecard.get("total_questions", len(qa_pairs)),
        scorecard.get("technical_score", 0),
        scorecard.get("communication_score", 0),
        scorecard.get("relevance_score", 0),
        scorecard.get("completeness_score", 0),
        scorecard.get("overall_score", 0),
        scorecard.get("interview_pct", 0),
        scorecard.get("resume_score", 0),
        scorecard.get("recommendation", ""),
        scorecard.get("status", ""),
        json.dumps(scorecard.get("top_strengths", [])),
        json.dumps(scorecard.get("top_improvements", [])),
        json.dumps(scorecard.get("category_scores", {})),
        json.dumps(scorecard.get("difficulty_scores", {})),
    ))
    iid = c.lastrowid

    for i, qa in enumerate(qa_pairs):
        c.execute("""
            INSERT INTO interview_qa (
                interview_id, question_no, question, category, difficulty, answer,
                technical_score, communication_score, relevance_score,
                completeness_score, overall_score, strengths, improvements,
                feedback, verdict
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            iid, i + 1,
            qa.get("question", ""),
            qa.get("category", ""),
            qa.get("difficulty", ""),
            qa.get("answer", ""),
            qa.get("technical_score", 0),
            qa.get("communication_score", 0),
            qa.get("relevance_score", 0),
            qa.get("completeness_score", 0),
            qa.get("overall_score", 0),
            json.dumps(qa.get("strengths", [])),
            json.dumps(qa.get("improvements", [])),
            qa.get("feedback", ""),
            qa.get("verdict", ""),
        ))

    conn.commit()
    conn.close()
    return iid


def get_all_interviews() -> list:
    """Return all interviews as list of dicts."""
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM interviews ORDER BY interview_date DESC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for json_field in ["top_strengths", "top_improvements",
                            "category_scores", "difficulty_scores"]:
            try:
                d[json_field] = json.loads(d[json_field] or "[]")
            except Exception:
                d[json_field] = []
        result.append(d)
    return result


def get_interview_qa(interview_id: int) -> list:
    """Return all Q&A rows for a specific interview."""
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM interview_qa WHERE interview_id=? ORDER BY question_no",
        (interview_id,)
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for json_field in ["strengths", "improvements"]:
            try:
                d[json_field] = json.loads(d[json_field] or "[]")
            except Exception:
                d[json_field] = []
        result.append(d)
    return result


def get_interview_by_id(interview_id: int) -> Optional[dict]:
    """Fetch one interview record."""
    init_db()
    conn = _connect()
    row  = conn.execute(
        "SELECT * FROM interviews WHERE id=?", (interview_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for f in ["top_strengths", "top_improvements", "category_scores", "difficulty_scores"]:
        try:
            d[f] = json.loads(d[f] or "[]")
        except Exception:
            d[f] = []
    return d


def delete_interview(interview_id: int):
    """Delete an interview and its Q&A rows."""
    init_db()
    conn = _connect()
    conn.execute("DELETE FROM interviews WHERE id=?", (interview_id,))
    conn.execute("DELETE FROM interview_qa WHERE interview_id=?", (interview_id,))
    conn.commit()
    conn.close()


def get_analytics() -> dict:
    """Compute aggregate analytics across all interviews."""
    init_db()
    conn = _connect()

    rows = conn.execute("SELECT * FROM interviews").fetchall()
    if not rows:
        conn.close()
        return {}

    interviews = [dict(r) for r in rows]

    # Basic stats
    scores  = [r["overall_score"] for r in interviews if r["overall_score"]]
    n       = len(interviews)
    avg     = round(sum(scores) / n, 1) if scores else 0
    best    = max(scores) if scores else 0

    # Role-wise
    role_data = {}
    for r in interviews:
        role = r["role"] or "Unknown"
        role_data.setdefault(role, []).append(r["overall_score"] or 0)
    role_avg = {k: round(sum(v)/len(v), 1) for k, v in role_data.items()}

    # Status counts
    status_counts = {}
    for r in interviews:
        s = r["status"] or "Unknown"
        status_counts[s] = status_counts.get(s, 0) + 1

    # Recommendation counts
    rec_counts = {}
    for r in interviews:
        rec = r["recommendation"] or "Unknown"
        rec_counts[rec] = rec_counts.get(rec, 0) + 1

    # Top candidates
    sorted_interviews = sorted(interviews, key=lambda x: x["overall_score"] or 0, reverse=True)

    # Monthly trend
    monthly = {}
    for r in interviews:
        try:
            month = r["interview_date"][:7]  # "YYYY-MM"
        except Exception:
            month = "Unknown"
        monthly.setdefault(month, []).append(r["overall_score"] or 0)
    monthly_avg = {k: round(sum(v)/len(v), 1) for k, v in sorted(monthly.items())}

    # Dimension averages
    dim_avgs = {
        "Technical":     round(sum(r["technical_score"] or 0 for r in interviews) / n, 1),
        "Communication": round(sum(r["communication_score"] or 0 for r in interviews) / n, 1),
        "Relevance":     round(sum(r["relevance_score"] or 0 for r in interviews) / n, 1),
        "Completeness":  round(sum(r["completeness_score"] or 0 for r in interviews) / n, 1),
    }

    # Success rate (Selected)
    success_rate = round(status_counts.get("Selected", 0) / n * 100, 1) if n else 0

    conn.close()
    return {
        "total_interviews":  n,
        "avg_score":         avg,
        "best_score":        best,
        "role_avg":          role_avg,
        "status_counts":     status_counts,
        "rec_counts":        rec_counts,
        "top_candidates":    sorted_interviews[:10],
        "monthly_trend":     monthly_avg,
        "dimension_avgs":    dim_avgs,
        "success_rate":      success_rate,
    }
