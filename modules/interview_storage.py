"""
modules/interview_storage.py
SQLite persistence layer for Phase 5 voice interviews.
Separate DB from Phase 4 (voice_interviews.db) so they coexist.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join("database", "voice_interviews.db")


def _connect():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_voice_db():
    """Create all tables if they don't exist."""
    conn = _connect()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_interviews (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name        TEXT    NOT NULL,
            candidate_id          TEXT,
            email                 TEXT,
            role                  TEXT,
            experience            TEXT,
            interview_date        TEXT,
            total_questions       INTEGER,
            -- Scores
            technical_score       REAL,
            communication_score   REAL,
            fluency_score         REAL,
            confidence_score      REAL,
            delivery_score        REAL,
            overall_score         REAL,
            voice_pct             REAL,
            composite_score       REAL,
            -- External scores
            resume_score          REAL DEFAULT 0,
            text_interview_score  REAL DEFAULT 0,
            -- Recommendation
            recommendation        TEXT,
            status                TEXT,
            -- Audio stats
            avg_wpm               REAL DEFAULT 0,
            avg_fillers           REAL DEFAULT 0,
            total_duration_sec    REAL DEFAULT 0,
            -- JSON blobs
            top_strengths         TEXT,
            top_improvements      TEXT,
            score_breakdown       TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_qa (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id          INTEGER REFERENCES voice_interviews(id) ON DELETE CASCADE,
            question_no           INTEGER,
            question              TEXT,
            category              TEXT,
            difficulty            TEXT,
            transcript            TEXT,
            audio_filename        TEXT,
            duration_sec          REAL DEFAULT 0,
            wpm                   REAL DEFAULT 0,
            filler_count          INTEGER DEFAULT 0,
            -- Dimension scores
            technical_score       REAL,
            communication_score   REAL,
            fluency_score         REAL,
            confidence_score      REAL,
            delivery_score        REAL,
            overall_score         REAL,
            verdict               TEXT,
            strengths             TEXT,
            improvements          TEXT,
            feedback              TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_voice_interview(
    candidate_name:        str,
    candidate_id:          str,
    email:                 str,
    role:                  str,
    experience:            str,
    scorecard:             dict,
    qa_pairs:              list,
) -> int:
    """
    Persist a complete voice interview session.
    Returns the new interview_id.
    """
    init_voice_db()
    conn = _connect()
    c    = conn.cursor()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")

    c.execute("""
        INSERT INTO voice_interviews (
            candidate_name, candidate_id, email, role, experience, interview_date,
            total_questions, technical_score, communication_score,
            fluency_score, confidence_score, delivery_score,
            overall_score, voice_pct, composite_score,
            resume_score, text_interview_score,
            recommendation, status,
            avg_wpm, avg_fillers, total_duration_sec,
            top_strengths, top_improvements, score_breakdown
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        candidate_name, candidate_id, email, role, experience, now,
        scorecard.get("total_questions", len(qa_pairs)),
        scorecard.get("technical_score",     0),
        scorecard.get("communication_score", 0),
        scorecard.get("fluency_score",       0),
        scorecard.get("confidence_score",    0),
        scorecard.get("delivery_score",      0),
        scorecard.get("overall_score",       0),
        scorecard.get("voice_pct",           0),
        scorecard.get("composite_score",     0),
        scorecard.get("resume_score",        0),
        scorecard.get("text_interview_score",0),
        scorecard.get("recommendation",      ""),
        scorecard.get("status",              ""),
        scorecard.get("avg_wpm",             0),
        scorecard.get("avg_fillers",         0),
        scorecard.get("total_duration_sec",  0),
        json.dumps(scorecard.get("top_strengths",  [])),
        json.dumps(scorecard.get("top_improvements",[])),
        json.dumps(scorecard.get("score_breakdown", [])),
    ))
    iid = c.lastrowid

    for i, qa in enumerate(qa_pairs):
        c.execute("""
            INSERT INTO voice_qa (
                interview_id, question_no, question, category, difficulty,
                transcript, audio_filename, duration_sec, wpm, filler_count,
                technical_score, communication_score, fluency_score,
                confidence_score, delivery_score, overall_score,
                verdict, strengths, improvements, feedback
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            iid, i + 1,
            qa.get("question",     ""),
            qa.get("category",     ""),
            qa.get("difficulty",   ""),
            qa.get("transcript",   ""),
            qa.get("audio_filename",""),
            qa.get("duration_sec", 0),
            qa.get("wpm",          0),
            qa.get("filler_count", 0),
            qa.get("technical_score",     0),
            qa.get("communication_score", 0),
            qa.get("fluency_score",       0),
            qa.get("confidence_score",    0),
            qa.get("delivery_score",      0),
            qa.get("overall_score",       0),
            qa.get("verdict",      ""),
            json.dumps(qa.get("strengths",    [])),
            json.dumps(qa.get("improvements", [])),
            qa.get("feedback",     ""),
        ))

    conn.commit()
    conn.close()
    return iid


def get_all_voice_interviews() -> list:
    """Return all voice interviews, newest first."""
    init_voice_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM voice_interviews ORDER BY interview_date DESC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for f in ["top_strengths", "top_improvements", "score_breakdown"]:
            try:
                d[f] = json.loads(d[f] or "[]")
            except Exception:
                d[f] = []
        result.append(d)
    return result


def get_voice_qa(interview_id: int) -> list:
    """Return all Q&A rows for one voice interview."""
    init_voice_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM voice_qa WHERE interview_id=? ORDER BY question_no",
        (interview_id,),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for f in ["strengths", "improvements"]:
            try:
                d[f] = json.loads(d[f] or "[]")
            except Exception:
                d[f] = []
        result.append(d)
    return result


def get_voice_interview_by_id(iid: int) -> Optional[dict]:
    init_voice_db()
    conn = _connect()
    row  = conn.execute(
        "SELECT * FROM voice_interviews WHERE id=?", (iid,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for f in ["top_strengths", "top_improvements", "score_breakdown"]:
        try:
            d[f] = json.loads(d[f] or "[]")
        except Exception:
            d[f] = []
    return d


def delete_voice_interview(iid: int):
    init_voice_db()
    conn = _connect()
    conn.execute("DELETE FROM voice_interviews WHERE id=?", (iid,))
    conn.execute("DELETE FROM voice_qa WHERE interview_id=?", (iid,))
    conn.commit()
    conn.close()


def get_voice_analytics() -> dict:
    """Compute aggregate analytics across all voice interviews."""
    init_voice_db()
    conn  = _connect()
    rows  = conn.execute("SELECT * FROM voice_interviews").fetchall()
    if not rows:
        conn.close()
        return {}

    interviews = [dict(r) for r in rows]
    n          = len(interviews)

    scores     = [r["overall_score"]   for r in interviews if r["overall_score"]]
    composites = [r["composite_score"] for r in interviews if r["composite_score"]]
    wpms       = [r["avg_wpm"]         for r in interviews if r["avg_wpm"]]

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    role_data = {}
    for r in interviews:
        role = r["role"] or "Unknown"
        role_data.setdefault(role, []).append(r["overall_score"] or 0)
    role_avg = {k: round(sum(v)/len(v), 1) for k, v in role_data.items()}

    status_counts = {}
    for r in interviews:
        s = r["status"] or "Unknown"
        status_counts[s] = status_counts.get(s, 0) + 1

    rec_counts = {}
    for r in interviews:
        rec = r["recommendation"] or "Unknown"
        rec_counts[rec] = rec_counts.get(rec, 0) + 1

    monthly = {}
    for r in interviews:
        try:
            month = r["interview_date"][:7]
        except Exception:
            month = "Unknown"
        monthly.setdefault(month, []).append(r["overall_score"] or 0)
    monthly_avg = {k: round(sum(v)/len(v), 1) for k, v in sorted(monthly.items())}

    dim_avgs = {
        "Technical":     safe_avg([r["technical_score"]     for r in interviews]),
        "Communication": safe_avg([r["communication_score"] for r in interviews]),
        "Fluency":       safe_avg([r["fluency_score"]       for r in interviews]),
        "Confidence":    safe_avg([r["confidence_score"]    for r in interviews]),
        "Delivery":      safe_avg([r["delivery_score"]      for r in interviews]),
    }

    conn.close()
    return {
        "total_interviews": n,
        "avg_score":        safe_avg(scores),
        "avg_composite":    safe_avg(composites),
        "avg_wpm":          safe_avg(wpms),
        "role_avg":         role_avg,
        "status_counts":    status_counts,
        "rec_counts":       rec_counts,
        "monthly_trend":    monthly_avg,
        "dimension_avgs":   dim_avgs,
        "top_candidates":   sorted(interviews,
                                   key=lambda x: x["overall_score"] or 0,
                                   reverse=True)[:10],
        "success_rate":     round(status_counts.get("Selected", 0) / n * 100, 1),
    }
