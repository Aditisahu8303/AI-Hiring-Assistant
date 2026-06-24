"""
modules/decision_engine.py
Calculate final hiring scores and AI decisions combining all phases.
"""

from collections import Counter
from typing import Optional

# ── Score weights ─────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "resume_score":     0.30,
    "interview_score":  0.25,
    "voice_score":      0.20,
    "behavioral_score": 0.15,
    "fraud_penalty":    0.10,  # subtracted
}

# ── Recommendation thresholds ─────────────────────────────────────────────
RECOMMENDATION_THRESHOLDS = {
    "Strong Hire": 80,
    "Hire":        65,
    "Consider":    50,
    "Reject":       0,
}

RECOMMENDATION_COLORS = {
    "Strong Hire": "#27AE60",
    "Hire":        "#2980B9",
    "Consider":    "#F39C12",
    "Reject":      "#E74C3C",
}


# ── Core scoring ──────────────────────────────────────────────────────────
def calculate_final_score(
    resume_score:     float | dict | None = None,
    interview_score:  float | None = None,
    voice_score:      float | None = None,
    behavioral_score: float | None = None,
    fraud_score:      float | None = None,
) -> dict:
    """
    Formula:
      raw     = resume*0.30 + interview*0.25 + voice*0.20 + behavioral*0.15
      penalty = fraud_score * 0.10
      final   = max(0, min(100, raw - penalty))

    Returns:
    {
      "final_score":              float,
      "raw_score":                float,
      "fraud_penalty_applied":    float,
      "component_scores":         dict,
      "recommendation":           str,
      "status_color":             str,
      "reasoning":                str,
      "confidence":               str,
    }
    """
    if isinstance(resume_score, dict):
        payload = resume_score
        resume_score     = payload.get("resume_score", payload.get("resume", 0))
        interview_score  = payload.get("interview_score", payload.get("interview", 0))
        voice_score      = payload.get("voice_score", payload.get("voice", 0))
        behavioral_score = payload.get("behavioral_score", payload.get("behavior", 0))
        fraud_score      = payload.get("fraud_score", payload.get("fraud", 0))

    if resume_score is None:
        resume_score = 0
    if interview_score is None:
        interview_score = 0
    if voice_score is None:
        voice_score = 0
    if behavioral_score is None:
        behavioral_score = 0
    if fraud_score is None:
        fraud_score = 0

    # Clamp all inputs to 0-100
    resume_score     = max(0.0, min(100.0, float(resume_score)))
    interview_score  = max(0.0, min(100.0, float(interview_score)))
    voice_score      = max(0.0, min(100.0, float(voice_score)))
    behavioral_score = max(0.0, min(100.0, float(behavioral_score)))
    fraud_score      = max(0.0, min(100.0, float(fraud_score)))

    raw_score = (
        resume_score     * SCORE_WEIGHTS["resume_score"]
        + interview_score  * SCORE_WEIGHTS["interview_score"]
        + voice_score      * SCORE_WEIGHTS["voice_score"]
        + behavioral_score * SCORE_WEIGHTS["behavioral_score"]
    )

    penalty = fraud_score * SCORE_WEIGHTS["fraud_penalty"]
    final_score = max(0.0, min(100.0, raw_score - penalty))

    recommendation = _get_recommendation(final_score)
    status_color   = RECOMMENDATION_COLORS[recommendation]

    # Confidence: High if most components are non-zero; Low if mostly zero
    non_zero = sum(1 for s in [resume_score, interview_score, voice_score, behavioral_score]
                   if s > 0)
    if non_zero >= 3:
        confidence = "High"
    elif non_zero == 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    reasoning = generate_reasoning(
        final_score, resume_score, interview_score, voice_score,
        behavioral_score, fraud_score, recommendation,
    )

    return {
        "final_score":           round(final_score, 2),
        "raw_score":             round(raw_score,   2),
        "fraud_penalty_applied": round(penalty,     2),
        "component_scores": {
            "resume_score":     round(resume_score,     2),
            "interview_score":  round(interview_score,  2),
            "voice_score":      round(voice_score,      2),
            "behavioral_score": round(behavioral_score, 2),
            "fraud_score":      round(fraud_score,      2),
        },
        "recommendation": recommendation,
        "status_color":   status_color,
        "reasoning":      reasoning,
        "confidence":     confidence,
    }


def _get_recommendation(final_score: float) -> str:
    if final_score >= RECOMMENDATION_THRESHOLDS["Strong Hire"]:
        return "Strong Hire"
    if final_score >= RECOMMENDATION_THRESHOLDS["Hire"]:
        return "Hire"
    if final_score >= RECOMMENDATION_THRESHOLDS["Consider"]:
        return "Consider"
    return "Reject"


# ── Reasoning generator ───────────────────────────────────────────────────
def generate_reasoning(
    final_score:      float | dict | None = None,
    resume_score:     float | None = None,
    interview_score:  float | None = None,
    voice_score:      float | None = None,
    behavioral_score: float | None = None,
    fraud_score:      float | None = None,
    recommendation:   str | None = None,
) -> str:
    """Generate a 2-3 sentence human-readable explanation of the recommendation."""
    if isinstance(final_score, dict):
        payload = final_score
        resume_score     = payload.get("resume_score", payload.get("resume", 0))
        interview_score  = payload.get("interview_score", payload.get("interview", 0))
        voice_score      = payload.get("voice_score", payload.get("voice", 0))
        behavioral_score = payload.get("behavioral_score", payload.get("behavior", 0))
        fraud_score      = payload.get("fraud_score", payload.get("fraud", 0))
        recommendation   = payload.get("recommendation", None)
        final_score      = payload.get("final_score", None)

    if resume_score is None:
        resume_score = 0
    if interview_score is None:
        interview_score = 0
    if voice_score is None:
        voice_score = 0
    if behavioral_score is None:
        behavioral_score = 0
    if fraud_score is None:
        fraud_score = 0
    if final_score is None:
        final_score = calculate_final_score(
            resume_score=resume_score,
            interview_score=interview_score,
            voice_score=voice_score,
            behavioral_score=behavioral_score,
            fraud_score=fraud_score,
        )["final_score"]
    if recommendation is None:
        recommendation = _get_recommendation(final_score)

    sentences = []

    # Opening sentence
    if recommendation == "Strong Hire":
        sentences.append(
            f"The candidate achieved an outstanding overall score of {final_score:.1f}/100, "
            "demonstrating strong performance across all evaluation dimensions."
        )
    elif recommendation == "Hire":
        sentences.append(
            f"With an overall score of {final_score:.1f}/100, the candidate meets the "
            "key hiring criteria and is recommended for the role."
        )
    elif recommendation == "Consider":
        sentences.append(
            f"The candidate's overall score of {final_score:.1f}/100 is borderline; "
            "they show promise but have notable gaps in one or more areas."
        )
    else:
        sentences.append(
            f"The candidate scored {final_score:.1f}/100, which falls below the minimum "
            "threshold for this position and is not recommended for hire."
        )

    # Highlight strengths / weaknesses
    scores = {
        "resume":     resume_score,
        "interview":  interview_score,
        "voice":      voice_score,
        "behavioral": behavioral_score,
    }
    best_dim  = max(scores, key=scores.get)
    worst_dim = min(scores, key=scores.get)

    if scores[best_dim] > 0:
        sentences.append(
            f"Their strongest dimension is {best_dim} ({scores[best_dim]:.1f}/100)"
            + (
                f", while {worst_dim} ({scores[worst_dim]:.1f}/100) has room for improvement."
                if scores[worst_dim] < scores[best_dim] - 15 else "."
            )
        )

    # Fraud note
    if fraud_score >= 60:
        sentences.append(
            f"A fraud risk score of {fraud_score:.1f} triggered a penalty of "
            f"{fraud_score * SCORE_WEIGHTS['fraud_penalty']:.1f} points; "
            "background verification is strongly advised."
        )
    elif fraud_score >= 30:
        sentences.append(
            f"A moderate fraud risk score ({fraud_score:.1f}) was detected; "
            "some profile claims should be verified."
        )

    return " ".join(sentences[:3])


# ── Batch scoring ─────────────────────────────────────────────────────────
def batch_score_candidates(candidates: list) -> list:
    """
    Score a list of candidate dicts.
    Each dict may have any subset of:
      resume_score, interview_score, voice_score, behavioral_score, fraud_score.
    Returns list of dicts with added final_score, recommendation, etc.
    Sorted by final_score descending with rank field added.
    """
    results = []
    for candidate in candidates:
        scored = dict(candidate)  # copy original fields
        result = calculate_final_score(
            resume_score     = candidate.get("resume_score",     0),
            interview_score  = candidate.get("interview_score",  0),
            voice_score      = candidate.get("voice_score",      0),
            behavioral_score = candidate.get("behavioral_score", 0),
            fraud_score      = candidate.get("fraud_score",      0),
        )
        scored.update(result)
        results.append(scored)

    # Sort descending by final_score
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Add rank
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return results


# ── Executive summary ─────────────────────────────────────────────────────
def get_executive_summary(candidates: list) -> dict:
    """
    Returns:
    {
      "total": int,
      "strong_hire": int, "hire": int, "consider": int, "reject": int,
      "avg_final_score": float,
      "top_candidate": dict,
      "hiring_success_rate": float,
      "avg_experience": float,
      "top_skills": [(skill, count)],
      "role_performance": {role: avg_score},
      "quality_index": float,
    }
    """
    if not candidates:
        return {
            "total":                0,
            "strong_hire":          0,
            "hire":                 0,
            "consider":             0,
            "reject":               0,
            "avg_final_score":      0.0,
            "top_candidate":        {},
            "hiring_success_rate":  0.0,
            "avg_experience":       0.0,
            "top_skills":           [],
            "role_performance":     {},
            "quality_index":        0.0,
        }

    # Score candidates if not already scored
    scored_candidates = []
    for c in candidates:
        if "final_score" not in c:
            scored = dict(c)
            result = calculate_final_score(
                resume_score     = c.get("resume_score",     0),
                interview_score  = c.get("interview_score",  0),
                voice_score      = c.get("voice_score",      0),
                behavioral_score = c.get("behavioral_score", 0),
                fraud_score      = c.get("fraud_score",      0),
            )
            scored.update(result)
            scored_candidates.append(scored)
        else:
            scored_candidates.append(c)

    total = len(scored_candidates)

    # Recommendation counts
    rec_counts = Counter(c.get("recommendation", "Reject") for c in scored_candidates)
    strong_hire = rec_counts.get("Strong Hire", 0)
    hire        = rec_counts.get("Hire",        0)
    consider    = rec_counts.get("Consider",    0)
    reject      = rec_counts.get("Reject",      0)

    # Average final score
    final_scores = [c.get("final_score", 0) for c in scored_candidates]
    avg_final    = round(sum(final_scores) / total, 2) if total else 0.0

    # Top candidate
    top_candidate = max(scored_candidates, key=lambda x: x.get("final_score", 0))

    # Hiring success rate
    hired = strong_hire + hire
    hiring_success_rate = round(hired / total * 100, 2) if total else 0.0

    # Average experience
    exp_values = []
    for c in scored_candidates:
        exp = c.get("experience_years", c.get("experience", None))
        try:
            exp_values.append(float(exp))
        except (TypeError, ValueError):
            pass
    avg_experience = round(sum(exp_values) / len(exp_values), 2) if exp_values else 0.0

    # Top skills across all candidates
    all_skills = []
    for c in scored_candidates:
        skills_raw = c.get("skills", "")
        if isinstance(skills_raw, list):
            for s in skills_raw:
                if s:
                    all_skills.append(str(s).strip().lower())
        elif isinstance(skills_raw, str) and skills_raw:
            import re
            for s in re.split(r"[,;|\n]", skills_raw):
                s = s.strip().lower()
                if s:
                    all_skills.append(s)

    skill_counter = Counter(all_skills)
    top_skills    = skill_counter.most_common(10)

    # Role performance (avg final score per role)
    role_scores: dict = {}
    for c in scored_candidates:
        role = str(c.get("role", c.get("job_title", c.get("position", "Unknown"))))
        role_scores.setdefault(role, []).append(c.get("final_score", 0))
    role_performance = {
        role: round(sum(scores) / len(scores), 2)
        for role, scores in role_scores.items()
    }

    # Quality index: avg_final_score * hiring_success_rate / 100
    quality_index = round(avg_final * hiring_success_rate / 100, 2)

    return {
        "total":                total,
        "strong_hire":          strong_hire,
        "hire":                 hire,
        "consider":             consider,
        "reject":               reject,
        "avg_final_score":      avg_final,
        "top_candidate":        top_candidate,
        "hiring_success_rate":  hiring_success_rate,
        "avg_experience":       avg_experience,
        "top_skills":           top_skills,
        "role_performance":     role_performance,
        "quality_index":        quality_index,
    }
