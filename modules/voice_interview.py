"""
modules/voice_interview.py
Voice interview session manager.
Orchestrates: question generation → audio upload/transcription → scoring → storage.
"""

import os
import io
from typing import Optional

from modules.interview_generator import generate_questions
from modules.speech_to_text      import (
    transcribe,
    extract_audio_features,
    enrich_features_from_transcript,
    WHISPER_AVAILABLE,
    SR_AVAILABLE,
)
from modules.audio_evaluator     import score_voice_answer, aggregate_voice_scorecard
from modules.interview_storage   import (
    init_voice_db,
    save_voice_interview,
    get_all_voice_interviews,
    get_voice_qa,
    get_voice_analytics,
)

# ── Public API ─────────────────────────────────────────────────────────────

def check_stt_capabilities() -> dict:
    """Return available STT libraries and recommended install instructions."""
    status = {
        "whisper":           WHISPER_AVAILABLE,
        "speech_recognition": SR_AVAILABLE,
        "ready":             WHISPER_AVAILABLE or SR_AVAILABLE,
    }
    if not status["ready"]:
        status["install_hint"] = (
            "Install speech-to-text:\n"
            "  pip install openai-whisper         # Recommended (offline, accurate)\n"
            "  pip install SpeechRecognition pydub # Alternative (needs internet)"
        )
    else:
        active = "whisper" if WHISPER_AVAILABLE else "SpeechRecognition"
        status["active_engine"] = active
    return status


def process_audio_answer(
    audio_bytes:  bytes,
    audio_name:   str,
    question:     str,
    category:     str,
    difficulty:   str,
    role:         str,
    prefer_whisper: bool = True,
    whisper_model:  str  = "base",
    api_key:        Optional[str] = None,
) -> dict:
    """
    Full pipeline: audio → transcript → evaluation.

    Returns a merged dict with transcription + evaluation fields.
    """
    # Step 1: Transcribe
    stt_result = transcribe(
        audio_bytes     = audio_bytes,
        filename        = audio_name,
        prefer_whisper  = prefer_whisper,
        whisper_model   = whisper_model,
    )

    transcript = stt_result.get("text", "")

    # Step 2: Audio features
    features = extract_audio_features(audio_bytes, audio_name)
    features = enrich_features_from_transcript(features, transcript)
    features["duration_sec"] = features.get("duration_sec") or stt_result.get("duration", 0)

    # Step 3: Evaluate
    eval_result = score_voice_answer(
        question      = question,
        transcript    = transcript,
        audio_features= features,
        category      = category,
        difficulty    = difficulty,
        role          = role,
        api_key       = api_key,
    )

    return {
        # Transcription
        "transcript":    transcript,
        "stt_engine":    stt_result.get("engine", "none"),
        "stt_error":     stt_result.get("error"),
        "language":      stt_result.get("language", "en"),
        "audio_filename":audio_name,
        # Evaluation
        **eval_result,
    }


def build_voice_session(
    candidate_name:  str,
    candidate_id:    str,
    email:           str,
    role:            str,
    experience:      str,
    n_questions:     int,
    skills:          list,
    api_key:         Optional[str] = None,
) -> list:
    """Generate questions for a new voice session."""
    return generate_questions(
        role             = role,
        skills           = skills,
        experience_level = experience,
        total_questions  = n_questions,
        api_key          = api_key,
    )


def finalize_session(
    candidate_name:        str,
    candidate_id:          str,
    email:                 str,
    role:                  str,
    experience:            str,
    qa_results:            list,
    resume_score:          float = 0.0,
    text_interview_score:  float = 0.0,
) -> dict:
    """
    Aggregate all per-question results into a scorecard and save to DB.

    Args:
        qa_results: list of process_audio_answer() dicts, each also containing
                    "question", "category", "difficulty" keys.
    Returns:
        scorecard dict
    """
    init_voice_db()
    eval_dicts = [r for r in qa_results]   # already contain all score keys

    scorecard = aggregate_voice_scorecard(
        evaluations          = eval_dicts,
        resume_score         = resume_score,
        text_interview_score = text_interview_score,
    )

    # Build qa_pairs for storage
    qa_pairs = []
    for r in qa_results:
        qa_pairs.append({
            "question":           r.get("question",          ""),
            "category":           r.get("category",          ""),
            "difficulty":         r.get("difficulty",        ""),
            "transcript":         r.get("transcript",        ""),
            "audio_filename":     r.get("audio_filename",    ""),
            "duration_sec":       r.get("duration_sec",      0),
            "wpm":                r.get("wpm",               0),
            "filler_count":       r.get("filler_count",      0),
            "technical_score":    r.get("technical_score",   0),
            "communication_score":r.get("communication_score",0),
            "fluency_score":      r.get("fluency_score",     0),
            "confidence_score":   r.get("confidence_score",  0),
            "delivery_score":     r.get("delivery_score",    0),
            "overall_score":      r.get("overall_score",     0),
            "verdict":            r.get("verdict",           ""),
            "strengths":          r.get("strengths",         []),
            "improvements":       r.get("improvements",      []),
            "feedback":           r.get("feedback",          ""),
        })

    iid = save_voice_interview(
        candidate_name = candidate_name,
        candidate_id   = candidate_id,
        email          = email,
        role           = role,
        experience     = experience,
        scorecard      = scorecard,
        qa_pairs       = qa_pairs,
    )
    scorecard["db_id"] = iid
    return scorecard
