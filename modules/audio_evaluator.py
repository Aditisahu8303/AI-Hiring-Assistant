"""
modules/audio_evaluator.py
Voice interview answer evaluator.
Extends Phase 4 text-based evaluation with voice-specific signals:
  - Speaking rate (words/min)
  - Vocabulary richness
  - Filler word penalty
  - Response confidence (structural, length, keyword depth)
  - Fluency score

All scoring is 0–10. No external API required.
"""

import re
import random
from typing import Optional

# Reuse Phase 4 text evaluator
from modules.interview_evaluator import (
    evaluate_answer,
    aggregate_interview_scores,
    _generate_recommendation,
)

# ── Voice-specific scoring constants ──────────────────────────────────────
FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "sort of",
                "kind of", "literally", "right", "okay so"]

POSITIVE_OPENERS = [
    "great question", "that's a good", "so ", "firstly", "to begin",
    "in my experience", "for example", "specifically", "the key point",
]

# Ideal words-per-minute range for interviews
WPM_IDEAL_LOW  = 120
WPM_IDEAL_HIGH = 180


def score_voice_answer(
    question: str,
    transcript: str,
    audio_features: dict,
    category: str,
    difficulty: str,
    role: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Full evaluation of a voice answer.

    Args:
        transcript:     text from speech-to-text
        audio_features: dict from speech_to_text.enrich_features_from_transcript()

    Returns dict with all scores + voice-specific metrics.
    """
    if not transcript or len(transcript.strip()) < 5:
        return _empty_voice_result()

    # ── Base text evaluation (reuse Phase 4) ──
    base = evaluate_answer(
        question   = question,
        answer     = transcript,
        category   = category,
        difficulty = difficulty,
        role       = role,
        api_key    = api_key,
    )

    # ── Voice-specific scores ──
    fluency     = _score_fluency(transcript, audio_features)
    confidence  = _score_confidence(transcript, audio_features)
    delivery    = _score_delivery(audio_features)

    # Merge into result
    result = {**base}
    result["fluency_score"]    = round(fluency, 1)
    result["confidence_score"] = round(confidence, 1)
    result["delivery_score"]   = round(delivery, 1)

    # Recalculate overall including voice dimensions
    result["overall_score"] = round(
        0.35 * base["technical_score"]
      + 0.20 * base["communication_score"]
      + 0.15 * base["relevance_score"]
      + 0.10 * base["completeness_score"]
      + 0.10 * fluency
      + 0.10 * confidence,
        1,
    )

    # Audio feature passthrough
    result["audio_features"] = audio_features
    result["wpm"]            = audio_features.get("words_per_min", 0)
    result["filler_count"]   = audio_features.get("filler_count", 0)
    result["duration_sec"]   = audio_features.get("duration_sec", 0)

    # Updated verdict
    s = result["overall_score"]
    result["verdict"] = (
        "Excellent" if s >= 8.0 else
        "Good"      if s >= 6.0 else
        "Average"   if s >= 4.0 else
        "Poor"
    )

    # Augment feedback with voice notes
    voice_notes = _build_voice_notes(audio_features, fluency, confidence, delivery)
    if voice_notes:
        result["feedback"] = result["feedback"] + " " + voice_notes

    return result


def _score_fluency(transcript: str, features: dict) -> float:
    """Score speaking fluency based on filler word ratio and vocabulary richness."""
    words         = len(transcript.split())
    filler_count  = features.get("filler_count", 0)
    vocab_rich    = features.get("vocabulary_richness", 0.5)

    if words == 0:
        return 0.0

    filler_ratio  = filler_count / words
    filler_penalty= min(3.0, filler_ratio * 30)
    vocab_bonus   = min(2.0, vocab_rich * 4)

    fluency = 7.0 - filler_penalty + vocab_bonus

    # Sentence variety bonus
    sentences = len([s for s in re.split(r'[.!?]+', transcript) if s.strip()])
    if sentences >= 3:
        fluency += 0.5

    return max(0.0, min(10.0, fluency))


def _score_confidence(transcript: str, features: dict) -> float:
    """Estimate confidence from vocabulary, structure, and assertive language."""
    text_lower = transcript.lower()

    # Positive indicators
    assertive   = ["is", "will", "can", "does", "the key", "specifically",
                   "i have", "we used", "in my project", "i implemented"]
    hedging     = ["maybe", "might", "i think", "not sure", "possibly",
                   "probably", "i don't know", "i'm not sure", "i believe maybe"]

    assertive_hits = sum(1 for w in assertive if w in text_lower)
    hedging_hits   = sum(1 for w in hedging   if w in text_lower)

    score  = 6.0
    score += min(2.0,  assertive_hits * 0.4)
    score -= min(2.5,  hedging_hits   * 0.5)

    # Word count confidence: more complete answers signal confidence
    words = features.get("word_count", 0)
    if words >= 120: score += 1.0
    elif words >= 60: score += 0.5
    elif words < 20:  score -= 1.5

    # Vocabulary richness
    vr = features.get("vocabulary_richness", 0.5)
    score += min(1.5, vr * 2)

    return max(0.0, min(10.0, score))


def _score_delivery(features: dict) -> float:
    """Score speaking delivery based on pace (words-per-minute)."""
    wpm = features.get("words_per_min", 0)
    dur = features.get("duration_sec", 0)

    if wpm == 0 or dur == 0:
        return 6.0   # neutral if no audio data

    # Ideal range: 120-180 wpm
    if WPM_IDEAL_LOW <= wpm <= WPM_IDEAL_HIGH:
        pace_score = 10.0
    elif wpm < 80:
        pace_score = max(3.0, 7.0 - (80 - wpm) * 0.05)   # too slow
    elif wpm > 220:
        pace_score = max(3.0, 7.0 - (wpm - 220) * 0.04)  # too fast
    else:
        # Gradual degradation outside ideal range
        dist = min(abs(wpm - WPM_IDEAL_LOW), abs(wpm - WPM_IDEAL_HIGH))
        pace_score = max(5.0, 10.0 - dist * 0.06)

    # Duration completeness
    if dur < 10:
        pace_score -= 2.0
    elif dur < 20:
        pace_score -= 1.0

    return max(0.0, min(10.0, pace_score))


def _build_voice_notes(features: dict, fluency: float,
                        confidence: float, delivery: float) -> str:
    notes = []
    wpm   = features.get("words_per_min", 0)
    fc    = features.get("filler_count",  0)
    dur   = features.get("duration_sec",  0)

    if fc >= 5:
        notes.append(f"Reduce filler words (detected ~{fc}).")
    if wpm > 0 and wpm < WPM_IDEAL_LOW:
        notes.append(f"Speaking pace is slow ({wpm:.0f} wpm) — aim for {WPM_IDEAL_LOW}-{WPM_IDEAL_HIGH} wpm.")
    if wpm > WPM_IDEAL_HIGH:
        notes.append(f"Speaking pace is fast ({wpm:.0f} wpm) — slow down for clarity.")
    if dur > 0 and dur < 15:
        notes.append("Answer is too brief — expand with examples and details.")
    if confidence < 5:
        notes.append("Use more assertive language to project confidence.")
    if fluency >= 8:
        notes.append("Excellent fluency — very natural delivery.")

    return " ".join(notes)


def _empty_voice_result() -> dict:
    return {
        "technical_score":     0.0,
        "communication_score": 0.0,
        "relevance_score":     0.0,
        "completeness_score":  0.0,
        "fluency_score":       0.0,
        "confidence_score":    0.0,
        "delivery_score":      0.0,
        "overall_score":       0.0,
        "strengths":           [],
        "improvements":        ["Please provide a spoken answer."],
        "feedback":            "No transcript detected.",
        "verdict":             "Poor",
        "audio_features":      {},
        "wpm":                 0,
        "filler_count":        0,
        "duration_sec":        0,
    }


# ── Interview-level aggregation ────────────────────────────────────────────
def aggregate_voice_scorecard(
    evaluations: list,
    resume_score: float = 0.0,
    text_interview_score: float = 0.0,
) -> dict:
    """
    Build the final voice interview scorecard.

    Args:
        evaluations:          list of score_voice_answer() dicts
        resume_score:         from Phase 1-3 (0-100)
        text_interview_score: from Phase 4 chat interview (0-100)
    """
    if not evaluations:
        return {}

    n = len(evaluations)

    def avg(key):
        return round(sum(e.get(key, 0) for e in evaluations) / n, 1)

    tech_avg    = avg("technical_score")
    comm_avg    = avg("communication_score")
    rel_avg     = avg("relevance_score")
    comp_avg    = avg("completeness_score")
    fluency_avg = avg("fluency_score")
    conf_avg    = avg("confidence_score")
    deliv_avg   = avg("delivery_score")
    overall_avg = avg("overall_score")

    voice_pct   = round(overall_avg * 10, 1)

    # Composite final score
    weights = {}
    composite_parts = []
    if resume_score > 0:
        composite_parts.append(("Resume", resume_score,       0.25))
    if text_interview_score > 0:
        composite_parts.append(("Chat Interview", text_interview_score, 0.35))
    composite_parts.append(("Voice Interview", voice_pct, 0.40 if composite_parts else 1.0))

    # Normalise weights
    total_w   = sum(w for _, _, w in composite_parts)
    composite = round(sum(s * w / total_w for _, s, w in composite_parts), 1)

    rec, status = _generate_recommendation(composite, resume_score, tech_avg)

    # Aggregate strengths and improvements
    all_s, all_i = [], []
    for e in evaluations:
        all_s.extend(e.get("strengths", []))
        all_i.extend(e.get("improvements", []))
    seen_s, seen_i = set(), set()
    top_s = [s for s in all_s if not (s in seen_s or seen_s.add(s))][:5]
    top_i = [i for i in all_i if not (i in seen_i or seen_i.add(i))][:5]

    # Audio stats
    avg_wpm      = round(sum(e.get("wpm", 0) for e in evaluations) / n, 1)
    avg_fillers  = round(sum(e.get("filler_count", 0) for e in evaluations) / n, 1)
    total_dur    = round(sum(e.get("duration_sec", 0) for e in evaluations), 1)

    return {
        "technical_score":     tech_avg,
        "communication_score": comm_avg,
        "relevance_score":     rel_avg,
        "completeness_score":  comp_avg,
        "fluency_score":       fluency_avg,
        "confidence_score":    conf_avg,
        "delivery_score":      deliv_avg,
        "overall_score":       overall_avg,
        "voice_pct":           voice_pct,
        "composite_score":     composite,
        "resume_score":        resume_score,
        "text_interview_score":text_interview_score,
        "recommendation":      rec,
        "status":              status,
        "total_questions":     n,
        "top_strengths":       top_s,
        "top_improvements":    top_i,
        "avg_wpm":             avg_wpm,
        "avg_fillers":         avg_fillers,
        "total_duration_sec":  total_dur,
        "score_breakdown":     composite_parts,
    }
