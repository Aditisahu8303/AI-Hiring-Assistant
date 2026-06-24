"""
modules/fraud_detection.py
Detect suspicious / fraudulent candidate applications from a pandas DataFrame.
"""

import re
from collections import Counter
from typing import Optional

import pandas as pd

# ── Comprehensive skills list for keyword stuffing detection ─────────────
_SKILL_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust",
    "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "node", "django", "flask",
    "fastapi", "spring", "tensorflow", "pytorch", "keras", "scikit", "sklearn",
    "pandas", "numpy", "matplotlib", "seaborn", "opencv", "nlp", "bert",
    "transformer", "gpt", "llm", "aws", "azure", "gcp", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "git", "linux", "bash", "powershell",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka",
    "spark", "hadoop", "airflow", "dbt", "tableau", "powerbi", "excel",
    "figma", "photoshop", "machine learning", "deep learning", "data science",
    "devops", "mlops", "agile", "scrum", "jira", "confluence",
}

FRESHER_KEYWORDS = {
    "fresher", "fresh graduate", "recent graduate", "entry level",
    "entry-level", "no experience", "0 years", "just graduated",
    "pursuing", "studying", "undergraduate", "btech", "b.tech",
    "bachelor", "bachelor's", "final year", "intern",
}

FRAUD_RULES = [
    "duplicate_name",
    "duplicate_email",
    "duplicate_phone",
    "unrealistic_exp",
    "exp_education_gap",
    "skill_inflation",
    "keyword_stuffing",
    "inconsistent_score",
    "generic_email",
]

RISK_THRESHOLDS = {"Low": (0, 30), "Medium": (30, 60), "High": (60, 101)}
RISK_COLORS     = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}


# ── Helper ─────────────────────────────────────────────────────────────────
def _get_str(candidate: dict, key: str, default: str = "") -> str:
    val = candidate.get(key, default)
    return str(val).strip() if val is not None else default


def _get_float(candidate: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(candidate.get(key, default))
    except (TypeError, ValueError):
        return default


def _count_unique_skills(skills_value) -> int:
    """Count unique skills from a comma/semicolon-separated string or list."""
    if skills_value is None:
        return 0
    if isinstance(skills_value, list):
        return len(set(s.strip().lower() for s in skills_value if s.strip()))
    skills_str = str(skills_value)
    parts = re.split(r"[,;|\n]", skills_str)
    return len(set(p.strip().lower() for p in parts if p.strip()))


def _risk_level(score: float) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


# ── Core scoring function ──────────────────────────────────────────────────
def calculate_fraud_score(
    candidate: dict,
    all_candidates: Optional[list] = None,
) -> dict:
    """
    Score one candidate for fraud risk.

    Returns:
    {
      "fraud_score": float,       # 0-100
      "risk_level": str,          # "Low" | "Medium" | "High"
      "risk_color": str,
      "flags": [str],
      "flag_count": int,
      "details": {rule_name: bool},
    }
    """
    score = 0.0
    flags = []
    details = {rule: False for rule in FRAUD_RULES}

    name  = _get_str(candidate, "candidate_name").lower()
    email = _get_str(candidate, "email").lower()
    phone = _get_str(candidate, "phone")
    exp   = _get_float(candidate, "experience_years", -1)
    resume_text = _get_str(candidate, "resume_text")
    education   = _get_str(candidate, "education").lower()
    match_score = _get_float(candidate, "match_score", -1)
    skills_raw  = candidate.get("skills", "")
    skill_count = _count_unique_skills(skills_raw)

    # ── 1. Duplicate name ──────────────────────────────────────────────
    if all_candidates and name:
        dup_count = sum(
            1 for c in all_candidates
            if _get_str(c, "candidate_name").lower() == name
        )
        if dup_count >= 2:
            score += 20
            flags.append("Duplicate name found in dataset.")
            details["duplicate_name"] = True

    # ── 2. Duplicate email ─────────────────────────────────────────────
    if all_candidates and email:
        dup_email = sum(
            1 for c in all_candidates
            if _get_str(c, "email").lower() == email
        )
        if dup_email >= 2:
            score += 25
            flags.append("Duplicate email address detected.")
            details["duplicate_email"] = True

    # ── 3. Duplicate phone ─────────────────────────────────────────────
    if all_candidates and phone:
        dup_phone = sum(
            1 for c in all_candidates
            if _get_str(c, "phone") == phone and phone != ""
        )
        if dup_phone >= 2:
            score += 20
            flags.append("Duplicate phone number detected.")
            details["duplicate_phone"] = True

    # ── 4. Unrealistic experience ──────────────────────────────────────
    if exp != -1:
        if exp > 40 or exp < 0:
            score += 15
            flags.append(
                f"Unrealistic experience value: {exp} years "
                f"({'negative' if exp < 0 else 'exceeds 40'})."
            )
            details["unrealistic_exp"] = True

    # ── 5. Experience / education gap ─────────────────────────────────
    if exp != -1 and exp >= 15 and education:
        is_fresher = any(kw in education for kw in FRESHER_KEYWORDS)
        if is_fresher:
            score += 10
            flags.append(
                "Claims 15+ years experience but education suggests fresher/entry-level."
            )
            details["exp_education_gap"] = True

    # ── 6. Skill inflation ─────────────────────────────────────────────
    if skill_count >= 20:
        score += 10
        flags.append(
            f"Skill inflation detected: {skill_count} unique skills listed (≥ 20)."
        )
        details["skill_inflation"] = True

    # ── 7. Keyword stuffing ────────────────────────────────────────────
    if resume_text:
        words = resume_text.lower().split()
        word_count = len(words)
        if word_count > 2000:
            found_skill_kws = sum(1 for kw in _SKILL_KEYWORDS if kw in resume_text.lower())
            if found_skill_kws > 30:
                score += 10
                flags.append(
                    f"Keyword stuffing: resume has {word_count} words and "
                    f"{found_skill_kws} technical keywords."
                )
                details["keyword_stuffing"] = True

    # ── 8. Inconsistent score ──────────────────────────────────────────
    if match_score != -1 and match_score > 95 and skill_count < 3:
        score += 15
        flags.append(
            f"Inconsistent profile: match score {match_score:.1f} but only {skill_count} skill(s) listed."
        )
        details["inconsistent_score"] = True

    # ── 9. Generic / suspicious email ─────────────────────────────────
    suspicious_email_terms = ["test", "fake", "dummy", "temp"]
    if email and any(term in email for term in suspicious_email_terms):
        score += 15
        flags.append(
            f"Suspicious email address contains generic/test keyword: '{email}'."
        )
        details["generic_email"] = True

    # Clamp score to 0-100
    final_score = min(100.0, max(0.0, score))
    risk = _risk_level(final_score)

    return {
        "fraud_score": round(final_score, 1),
        "risk_level":  risk,
        "risk_color":  RISK_COLORS[risk],
        "flags":       flags,
        "flag_count":  len(flags),
        "details":     details,
    }


# ── Dataset-level analysis ─────────────────────────────────────────────────
def analyze_dataset_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud detection on entire dataset.
    Adds columns: fraud_score, risk_level, risk_color, flag_count, fraud_flags
    Returns the enriched DataFrame.
    """
    records = df.to_dict(orient="records")

    fraud_scores  = []
    risk_levels   = []
    risk_colors   = []
    flag_counts   = []
    fraud_flags   = []

    for candidate in records:
        result = calculate_fraud_score(candidate, all_candidates=records)
        fraud_scores.append(result["fraud_score"])
        risk_levels.append(result["risk_level"])
        risk_colors.append(result["risk_color"])
        flag_counts.append(result["flag_count"])
        fraud_flags.append("; ".join(result["flags"]) if result["flags"] else "None")

    df = df.copy()
    df["fraud_score"] = fraud_scores
    df["risk_level"]  = risk_levels
    df["risk_color"]  = risk_colors
    df["flag_count"]  = flag_counts
    df["fraud_flags"] = fraud_flags

    return df


# ── Summary statistics ─────────────────────────────────────────────────────
def get_fraud_summary(df: pd.DataFrame) -> dict:
    """
    Returns:
    {
      "total": int,
      "low_risk": int, "medium_risk": int, "high_risk": int,
      "avg_fraud_score": float,
      "high_risk_candidates": [{"name": str, "score": float, "flags": [str]}],
      "most_common_flags": [(flag_name, count)],   # top 5
    }
    """
    if df.empty:
        return {
            "total": 0,
            "low_risk": 0,
            "medium_risk": 0,
            "high_risk": 0,
            "avg_fraud_score": 0.0,
            "high_risk_candidates": [],
            "most_common_flags": [],
        }

    # Ensure required columns exist (in case analyze_dataset_fraud wasn't called)
    records = df.to_dict(orient="records")
    if "fraud_score" not in df.columns:
        df = analyze_dataset_fraud(df)
        records = df.to_dict(orient="records")

    name_col = _detect_column(df, ["candidate_name", "name", "Name"])

    total        = len(df)
    low_risk     = int((df["risk_level"] == "Low").sum())
    medium_risk  = int((df["risk_level"] == "Medium").sum())
    high_risk    = int((df["risk_level"] == "High").sum())
    avg_score    = round(float(df["fraud_score"].mean()), 2)

    # High risk candidates details
    high_risk_df = df[df["risk_level"] == "High"].copy()
    high_risk_candidates = []
    for _, row in high_risk_df.iterrows():
        cname = str(row[name_col]) if name_col and name_col in row else "Unknown"
        flags_str = str(row.get("fraud_flags", ""))
        flags_list = [f.strip() for f in flags_str.split(";") if f.strip() and f.strip() != "None"]
        high_risk_candidates.append({
            "name":  cname,
            "score": round(float(row.get("fraud_score", 0)), 1),
            "flags": flags_list,
        })
    # Sort by score descending
    high_risk_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Most common individual flags
    all_flags = []
    for _, row in df.iterrows():
        flags_str = str(row.get("fraud_flags", ""))
        for f in flags_str.split(";"):
            f = f.strip()
            if f and f != "None":
                all_flags.append(f)

    flag_counter = Counter(all_flags)
    most_common_flags = flag_counter.most_common(5)

    return {
        "total":                 total,
        "low_risk":              low_risk,
        "medium_risk":           medium_risk,
        "high_risk":             high_risk,
        "avg_fraud_score":       avg_score,
        "high_risk_candidates":  high_risk_candidates,
        "most_common_flags":     most_common_flags,
    }


def _detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return first matching column name from a list of possibilities."""
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) > 0 else None
