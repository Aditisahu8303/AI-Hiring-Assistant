"""
modules/interview_evaluator.py
AI Answer Evaluator — scores candidate answers across multiple dimensions.
Uses Sentence Transformers (semantic similarity) when available,
with a robust keyword/heuristic fallback that requires no external API.
"""

import re
import math
import random
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ── Cached model ─────────────────────────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is None and ST_AVAILABLE:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
    return _model


# ── Ideal answer keywords (for heuristic scoring) ────────────────────────────
IDEAL_KEYWORDS = {
    "supervised":       ["labeled", "training data", "classification", "regression", "labels"],
    "unsupervised":     ["unlabeled", "clustering", "k-means", "pca", "dimensionality"],
    "overfitting":      ["regularization", "cross-validation", "dropout", "early stopping", "complexity"],
    "bias-variance":    ["tradeoff", "underfitting", "overfitting", "complexity", "generalization"],
    "random forest":    ["ensemble", "decision tree", "bagging", "feature importance", "bootstrap"],
    "xgboost":          ["gradient boosting", "sequential", "residuals", "learning rate", "trees"],
    "neural network":   ["layers", "weights", "activation", "backpropagation", "gradient"],
    "transformer":      ["attention", "self-attention", "encoder", "decoder", "positional encoding"],
    "precision recall": ["true positive", "false positive", "false negative", "threshold", "f1"],
    "sql":              ["query", "join", "select", "where", "aggregate", "index"],
    "docker":           ["container", "image", "dockerfile", "virtualization", "isolation"],
    "rest api":         ["http", "endpoint", "get", "post", "stateless", "json", "status code"],
    "git":              ["commit", "branch", "merge", "rebase", "repository", "pull request"],
    "microservices":    ["independent", "api", "scalable", "decoupled", "service", "deployment"],
    "python":           ["interpreter", "dynamic", "object-oriented", "library", "pip"],
}

# Communication quality signals
POSITIVE_SIGNALS  = [
    "for example", "such as", "specifically", "in my experience", "the reason",
    "because", "therefore", "however", "on the other hand", "in addition",
    "first", "second", "finally", "furthermore", "to summarize",
]
NEGATIVE_SIGNALS  = [
    "i don't know", "not sure", "no idea", "i have no", "i cannot answer",
    "i never", "skip", "pass", "n/a", "idk",
]
FILLER_WORDS      = ["um", "uh", "like", "you know", "basically", "literally", "sort of"]


# ── Main evaluation function ──────────────────────────────────────────────────
def evaluate_answer(
    question: str,
    answer: str,
    category: str,
    difficulty: str,
    role: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Evaluate a candidate's answer.

    Returns:
        {
          "technical_score":      float,  # 0-10
          "communication_score":  float,
          "relevance_score":      float,
          "completeness_score":   float,
          "overall_score":        float,
          "strengths":            [str],
          "improvements":         [str],
          "feedback":             str,
          "verdict":              str,   # Excellent / Good / Average / Poor
        }
    """
    if not answer or not answer.strip() or len(answer.strip()) < 5:
        return _empty_answer_result()

    # Try API evaluation first
    if api_key:
        result = _evaluate_via_api(question, answer, category, difficulty, role, api_key)
        if result:
            return result

    # Robust local evaluation
    return _evaluate_locally(question, answer, category, difficulty, role)


def _empty_answer_result() -> dict:
    return {
        "technical_score":     0.0,
        "communication_score": 0.0,
        "relevance_score":     0.0,
        "completeness_score":  0.0,
        "overall_score":       0.0,
        "strengths":           [],
        "improvements":        ["Please provide a substantive answer."],
        "feedback":            "No answer was provided.",
        "verdict":             "Poor",
    }


def _evaluate_via_api(question, answer, category, difficulty, role, api_key):
    """Try Gemini then OpenAI. Returns None on failure."""
    prompt = f"""You are an expert technical interviewer evaluating a candidate's interview answer.

Role: {role}
Category: {category}
Difficulty: {difficulty}
Question: {question}
Candidate Answer: {answer}

Evaluate the answer and respond with ONLY this JSON structure (no markdown, no extra text):
{{
  "technical_score": <0-10>,
  "communication_score": <0-10>,
  "relevance_score": <0-10>,
  "completeness_score": <0-10>,
  "overall_score": <0-10>,
  "strengths": ["strength1", "strength2"],
  "improvements": ["improvement1", "improvement2"],
  "feedback": "2-3 sentence detailed feedback",
  "verdict": "Excellent|Good|Average|Poor"
}}"""

    try:
        import google.generativeai as genai
        import json
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt)
        text  = resp.text.strip().strip("```json").strip("```").strip()
        data  = json.loads(text)
        return _validate_eval_result(data)
    except Exception:
        pass

    try:
        import openai, json
        client = openai.OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        text   = resp.choices[0].message.content.strip()
        data   = json.loads(text)
        return _validate_eval_result(data)
    except Exception:
        return None


def _validate_eval_result(data: dict) -> dict:
    """Clamp all scores to 0-10 and ensure required keys."""
    required = ["technical_score","communication_score","relevance_score",
                "completeness_score","overall_score","strengths","improvements",
                "feedback","verdict"]
    for key in required:
        if key not in data:
            return None
    for score_key in ["technical_score","communication_score","relevance_score",
                      "completeness_score","overall_score"]:
        try:
            data[score_key] = max(0.0, min(10.0, float(data[score_key])))
        except Exception:
            data[score_key] = 5.0
    return data


def _evaluate_locally(question: str, answer: str, category: str,
                      difficulty: str, role: str) -> dict:
    """
    Rule-based evaluation using:
    - Answer length & structure
    - Keyword matching (question + domain keywords)
    - Communication quality signals
    - Semantic similarity (if SentenceTransformer available)
    """
    answer_lower   = answer.lower().strip()
    answer_words   = answer_lower.split()
    word_count     = len(answer_words)
    sentence_count = len([s for s in re.split(r'[.!?]+', answer) if s.strip()])

    # ── Completeness: length-based ──────────────────────────────────────────
    diff_min_words = {"Easy": 30, "Medium": 60, "Hard": 100}.get(difficulty, 50)
    if word_count >= diff_min_words * 2:
        completeness = 9.5
    elif word_count >= diff_min_words:
        completeness = 7.5
    elif word_count >= diff_min_words * 0.5:
        completeness = 5.0
    else:
        completeness = max(1.0, word_count / diff_min_words * 5)

    # ── Technical accuracy: keyword matching ────────────────────────────────
    question_lower = question.lower()
    relevant_kws   = []
    for trigger, keywords in IDEAL_KEYWORDS.items():
        if trigger in question_lower or trigger in answer_lower:
            relevant_kws.extend(keywords)

    # Also extract nouns/terms from the question itself
    question_terms = [w for w in re.findall(r'\b[a-z]{4,}\b', question_lower)
                      if w not in {"what","when","where","explain","describe","how","would","should","could","which","there","their","about","with"}]
    relevant_kws.extend(question_terms)
    relevant_kws = list(set(relevant_kws))

    if relevant_kws:
        matched = sum(1 for kw in relevant_kws if kw in answer_lower)
        tech_raw = matched / len(relevant_kws)
    else:
        # Fallback: word overlap ratio
        q_words = set(re.findall(r'\b[a-z]{4,}\b', question_lower))
        a_words = set(re.findall(r'\b[a-z]{4,}\b', answer_lower))
        tech_raw = len(q_words & a_words) / max(len(q_words), 1)

    # Semantic boost if model available
    model = _get_model()
    sem_sim = 0.5
    if model:
        try:
            embs    = model.encode([question, answer])
            sem_sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        except Exception:
            sem_sim = 0.5

    # Blend: 40% keyword, 60% semantic (or 100% keyword if no model)
    if model:
        technical = min(10.0, (0.4 * tech_raw + 0.6 * sem_sim) * 12)
    else:
        technical = min(10.0, tech_raw * 10 + 1.5)  # slight bonus for any keywords

    # Hard questions need higher bar
    difficulty_penalty = {"Easy": 0, "Medium": -0.5, "Hard": -1.0}.get(difficulty, 0)
    technical = max(0.0, technical + difficulty_penalty)

    # ── Communication score ─────────────────────────────────────────────────
    pos_hits     = sum(1 for sig in POSITIVE_SIGNALS if sig in answer_lower)
    neg_hits     = sum(1 for sig in NEGATIVE_SIGNALS if sig in answer_lower)
    filler_hits  = sum(answer_words.count(fw) for fw in FILLER_WORDS)
    has_structure= any(marker in answer_lower for marker in
                       ["first", "second", "third", "1.", "2.", "•", "-", "finally"])

    comm_score  = 5.0
    comm_score += min(2.0, pos_hits * 0.5)
    comm_score -= min(3.0, neg_hits * 1.5)
    comm_score -= min(1.5, filler_hits * 0.3)
    comm_score += 1.0 if has_structure else 0
    comm_score += min(1.5, sentence_count * 0.1)
    comm_score  = max(0.0, min(10.0, comm_score))

    # ── Relevance ───────────────────────────────────────────────────────────
    if model:
        relevance = min(10.0, sem_sim * 10)
    else:
        # Check if answer contains question's key nouns
        key_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', question))
        key_nouns |= set(question_terms[:8])
        if key_nouns:
            rel_hits  = sum(1 for n in key_nouns if n.lower() in answer_lower)
            relevance = min(10.0, (rel_hits / len(key_nouns)) * 10 + 2.0)
        else:
            relevance = 6.0

    # ── Overall score ────────────────────────────────────────────────────────
    weights   = {"technical": 0.40, "communication": 0.25,
                 "relevance": 0.20, "completeness": 0.15}
    overall   = (technical    * weights["technical"]
               + comm_score   * weights["communication"]
               + relevance    * weights["relevance"]
               + completeness * weights["completeness"])
    overall   = round(max(0.0, min(10.0, overall)), 1)

    # ── Generate feedback ────────────────────────────────────────────────────
    strengths, improvements = _generate_feedback(
        answer, technical, comm_score, completeness, relevance,
        word_count, pos_hits, has_structure, role, category,
    )

    verdict = (
        "Excellent" if overall >= 8.0 else
        "Good"      if overall >= 6.0 else
        "Average"   if overall >= 4.0 else
        "Poor"
    )

    feedback_text = _build_feedback_text(
        overall, technical, comm_score, difficulty, category, verdict, role
    )

    return {
        "technical_score":     round(technical, 1),
        "communication_score": round(comm_score, 1),
        "relevance_score":     round(relevance, 1),
        "completeness_score":  round(completeness, 1),
        "overall_score":       overall,
        "strengths":           strengths,
        "improvements":        improvements,
        "feedback":            feedback_text,
        "verdict":             verdict,
    }


def _generate_feedback(answer, technical, comm, completeness, relevance,
                        word_count, pos_hits, has_structure, role, category):
    strengths, improvements = [], []

    if technical >= 7.0:
        strengths.append(f"Strong technical understanding demonstrated for a {role} role.")
    elif technical >= 5.0:
        strengths.append("Shows foundational technical knowledge.")

    if comm >= 7.0:
        strengths.append("Communicates ideas clearly and effectively.")
    if has_structure:
        strengths.append("Answer is well-structured with clear logical flow.")
    if pos_hits >= 2:
        strengths.append("Uses good examples and elaborates on key points.")
    if completeness >= 7.0:
        strengths.append("Provides a comprehensive and detailed response.")

    if word_count < 40:
        improvements.append("Expand your answer — provide more depth and detail.")
    if technical < 5.0:
        improvements.append(f"Improve technical depth for {category} questions.")
    if comm < 5.0:
        improvements.append("Structure your answer more clearly (use points/steps).")
    if relevance < 5.0:
        improvements.append("Ensure your answer directly addresses what was asked.")
    if not has_structure and word_count > 60:
        improvements.append("Use numbered points or sections to organize longer answers.")

    # Ensure at least one in each
    if not strengths:
        strengths.append("Made an attempt to answer the question.")
    if not improvements:
        improvements.append("Continue practicing to maintain this strong performance.")

    return strengths[:3], improvements[:3]


def _build_feedback_text(overall, technical, comm, difficulty, category, verdict, role):
    lines = []

    if overall >= 8.0:
        lines.append(f"Excellent response for a {role} position.")
    elif overall >= 6.0:
        lines.append(f"Good answer that covers the key concepts for this {category} question.")
    elif overall >= 4.0:
        lines.append(f"Average response — the answer addresses some aspects but lacks depth.")
    else:
        lines.append(f"This answer needs significant improvement for a {role} role.")

    if difficulty == "Hard" and overall >= 6.0:
        lines.append("Impressive performance on a challenging question.")
    elif difficulty == "Hard" and overall < 5.0:
        lines.append("Hard questions require deeper technical detail — consider revising this topic.")

    if technical > comm + 2:
        lines.append("Strong technical knowledge but focus on clearer communication.")
    elif comm > technical + 2:
        lines.append("Well communicated, but strengthen the technical accuracy of your answer.")

    return " ".join(lines)


# ── Interview-level aggregation ───────────────────────────────────────────────
def aggregate_interview_scores(evaluations: list, resume_score: float = 0.0) -> dict:
    """
    Aggregate individual question evaluations into a final interview scorecard.

    Args:
        evaluations: list of evaluate_answer() dicts
        resume_score: candidate's Phase 1-2 resume match score (0-100)

    Returns:
        Full scorecard dict
    """
    if not evaluations:
        return {}

    n = len(evaluations)

    def avg(key):
        return round(sum(e.get(key, 0) for e in evaluations) / n, 1)

    tech_avg  = avg("technical_score")
    comm_avg  = avg("communication_score")
    rel_avg   = avg("relevance_score")
    comp_avg  = avg("completeness_score")
    overall   = avg("overall_score")

    # Category breakdown
    by_category = {}
    for ev in evaluations:
        cat = ev.get("category", "General")
        by_category.setdefault(cat, []).append(ev["overall_score"])
    category_scores = {k: round(sum(v)/len(v), 1) for k, v in by_category.items()}

    # Difficulty breakdown
    by_diff = {}
    for ev in evaluations:
        d = ev.get("difficulty", "Medium")
        by_diff.setdefault(d, []).append(ev["overall_score"])
    difficulty_scores = {k: round(sum(v)/len(v), 1) for k, v in by_diff.items()}

    # Normalize to 100-point scale
    interview_pct = round(overall * 10, 1)

    # Final recommendation
    rec, status = _generate_recommendation(interview_pct, resume_score, tech_avg)

    # Strengths & improvements aggregation
    all_strengths    = []
    all_improvements = []
    for ev in evaluations:
        all_strengths.extend(ev.get("strengths", []))
        all_improvements.extend(ev.get("improvements", []))

    # Deduplicate (keep order)
    seen_s, seen_i = set(), set()
    unique_strengths    = [s for s in all_strengths    if not (s in seen_s or seen_s.add(s))][:5]
    unique_improvements = [i for i in all_improvements if not (i in seen_i or seen_i.add(i))][:5]

    return {
        "technical_score":     tech_avg,
        "communication_score": comm_avg,
        "relevance_score":     rel_avg,
        "completeness_score":  comp_avg,
        "overall_score":       overall,
        "interview_pct":       interview_pct,
        "resume_score":        resume_score,
        "category_scores":     category_scores,
        "difficulty_scores":   difficulty_scores,
        "total_questions":     n,
        "recommendation":      rec,
        "status":              status,
        "top_strengths":       unique_strengths,
        "top_improvements":    unique_improvements,
    }


def _generate_recommendation(interview_pct: float, resume_score: float, tech: float):
    """Generate final hiring recommendation."""
    # Weighted final score
    if resume_score > 0:
        final = 0.5 * interview_pct + 0.5 * resume_score
    else:
        final = interview_pct

    if final >= 82 and tech >= 7.5:
        return "Strong Hire", "Selected"
    elif final >= 68 and tech >= 6.0:
        return "Hire", "Selected"
    elif final >= 52:
        return "Consider", "Hold"
    else:
        return "Reject", "Rejected"
