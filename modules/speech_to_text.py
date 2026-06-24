"""
modules/speech_to_text.py
Speech-to-text conversion using OpenAI Whisper (primary)
with SpeechRecognition as fallback.
All functions return {"text": str, "language": str, "duration": float, "engine": str, "error": str|None}
"""

import os
import io
import wave
import time
import tempfile
from typing import Optional

# ── Library availability flags ─────────────────────────────────────────────
try:
    import whisper as _whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as _sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    from pydub import AudioSegment as _AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import soundfile as _sf
    import numpy as _np
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

# ── Whisper model cache ────────────────────────────────────────────────────
_whisper_model = None

def _get_whisper_model(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        try:
            _whisper_model = _whisper.load_model(model_size)
        except Exception:
            pass
    return _whisper_model


# ── Audio duration helper ──────────────────────────────────────────────────
def get_audio_duration(audio_bytes: bytes, filename: str = "audio.wav") -> float:
    """Estimate duration in seconds from raw audio bytes."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".wav":
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                frames = wf.getnframes()
                rate   = wf.getframerate()
                return round(frames / float(rate), 2) if rate else 0.0
        if PYDUB_AVAILABLE:
            fmt = ext.lstrip(".") or "mp3"
            seg = _AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            return round(len(seg) / 1000.0, 2)
        if SF_AVAILABLE:
            data, sr = _sf.read(io.BytesIO(audio_bytes))
            return round(len(data) / sr, 2) if sr else 0.0
    except Exception:
        pass
    return 0.0


def _save_to_temp(audio_bytes: bytes, suffix: str) -> str:
    """Write bytes to a named temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
    except Exception:
        os.close(fd)
    return path


def _convert_to_wav(audio_bytes: bytes, ext: str) -> Optional[bytes]:
    """Convert any audio format to WAV bytes using pydub."""
    if not PYDUB_AVAILABLE:
        return None
    try:
        fmt = ext.lstrip(".") or "mp3"
        seg = _AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return None


# ── Primary: Whisper ───────────────────────────────────────────────────────
def transcribe_with_whisper(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    model_size: str = "base",
) -> dict:
    """Transcribe audio bytes using OpenAI Whisper."""
    if not WHISPER_AVAILABLE:
        return _not_available("whisper")

    ext  = os.path.splitext(filename)[1].lower() or ".wav"
    model = _get_whisper_model(model_size)
    if model is None:
        return {"text": "", "language": "unknown", "duration": 0.0,
                "engine": "whisper", "error": "Could not load Whisper model."}

    # Convert to wav if needed
    if ext not in (".wav",):
        wav_bytes = _convert_to_wav(audio_bytes, ext)
        if wav_bytes:
            audio_bytes = wav_bytes
            ext = ".wav"

    tmp_path = _save_to_temp(audio_bytes, ext)
    try:
        t0     = time.time()
        result = model.transcribe(tmp_path, fp16=False)
        elapsed = round(time.time() - t0, 2)
        text   = result.get("text", "").strip()
        lang   = result.get("language", "en")
        dur    = get_audio_duration(audio_bytes, filename)
        return {
            "text":     text,
            "language": lang,
            "duration": dur,
            "engine":   f"whisper-{model_size}",
            "error":    None,
            "proc_time": elapsed,
        }
    except Exception as e:
        return {"text": "", "language": "unknown", "duration": 0.0,
                "engine": "whisper", "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Fallback: SpeechRecognition (Google) ──────────────────────────────────
def transcribe_with_sr(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """Transcribe audio using SpeechRecognition (requires internet for Google API)."""
    if not SR_AVAILABLE:
        return _not_available("SpeechRecognition")

    ext = os.path.splitext(filename)[1].lower() or ".wav"

    # Need WAV for SR
    if ext != ".wav":
        wav_bytes = _convert_to_wav(audio_bytes, ext)
        if wav_bytes:
            audio_bytes = wav_bytes
        else:
            return {"text": "", "language": "en", "duration": 0.0,
                    "engine": "SpeechRecognition", "error": "Cannot convert audio to WAV."}

    tmp_path = _save_to_temp(audio_bytes, ".wav")
    try:
        recognizer = _sr.Recognizer()
        with _sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        dur  = get_audio_duration(audio_bytes, filename)
        return {
            "text":     text,
            "language": "en",
            "duration": dur,
            "engine":   "google-sr",
            "error":    None,
        }
    except _sr.UnknownValueError:
        return {"text": "", "language": "en", "duration": 0.0,
                "engine": "google-sr", "error": "Speech not understood."}
    except Exception as e:
        return {"text": "", "language": "en", "duration": 0.0,
                "engine": "google-sr", "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Main dispatch ──────────────────────────────────────────────────────────
def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    prefer_whisper: bool = True,
    whisper_model: str = "base",
) -> dict:
    """
    Transcribe audio using the best available engine.
    Falls back gracefully if libraries are not installed.
    """
    if not audio_bytes:
        return {"text": "", "language": "unknown", "duration": 0.0,
                "engine": "none", "error": "No audio data provided."}

    if prefer_whisper and WHISPER_AVAILABLE:
        result = transcribe_with_whisper(audio_bytes, filename, whisper_model)
        if result["error"] is None and result["text"]:
            return result
        # Fall through to SR if whisper failed

    if SR_AVAILABLE:
        return transcribe_with_sr(audio_bytes, filename)

    # No audio library available — return instructive message
    return {
        "text":     "",
        "language": "unknown",
        "duration": get_audio_duration(audio_bytes, filename),
        "engine":   "none",
        "error":    (
            "No speech-to-text library installed.\n"
            "Install one of:\n"
            "  pip install openai-whisper\n"
            "  pip install SpeechRecognition pydub"
        ),
    }


def _not_available(lib: str) -> dict:
    return {
        "text":     "",
        "language": "unknown",
        "duration": 0.0,
        "engine":   lib,
        "error":    f"{lib} is not installed. Run: pip install openai-whisper",
    }


# ── Audio feature extraction (no extra deps needed beyond numpy) ───────────
def extract_audio_features(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Extract basic audio features for analytics.
    Returns speaking rate, estimated confidence signals, etc.
    """
    features = {
        "duration_sec":   0.0,
        "word_count":     0,
        "words_per_min":  0.0,
        "avg_word_len":   0.0,
        "sentence_count": 0,
        "filler_count":   0,
        "unique_words":   0,
        "vocabulary_richness": 0.0,
    }

    dur = get_audio_duration(audio_bytes, filename)
    features["duration_sec"] = dur
    return features


def enrich_features_from_transcript(features: dict, transcript: str) -> dict:
    """Add text-based features once transcript is available."""
    import re
    if not transcript:
        return features

    words     = transcript.lower().split()
    sentences = [s for s in re.split(r'[.!?]+', transcript) if s.strip()]
    fillers   = ["um", "uh", "like", "you know", "basically", "sort of", "kind of"]

    features["word_count"]     = len(words)
    features["sentence_count"] = len(sentences)
    features["filler_count"]   = sum(words.count(f) for f in fillers)
    features["unique_words"]   = len(set(words))
    features["avg_word_len"]   = round(
        sum(len(w) for w in words) / len(words), 2) if words else 0
    features["vocabulary_richness"] = round(
        features["unique_words"] / max(len(words), 1), 3)

    dur = features.get("duration_sec", 0)
    if dur > 0 and features["word_count"] > 0:
        features["words_per_min"] = round(features["word_count"] / dur * 60, 1)

    return features
