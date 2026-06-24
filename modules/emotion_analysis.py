"""
modules/emotion_analysis.py
Emotion detection from uploaded images/video frames.
Primary: DeepFace (if installed).
Fallback: Heuristic pixel-brightness + color analysis (numpy/Pillow).
"""

import io
import math
import random
from typing import Optional

# ── Optional imports with graceful fallback ────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────
EMOTIONS = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"]

EMOTION_COLORS = {
    "happy":    "#FFD700",
    "neutral":  "#87CEEB",
    "sad":      "#4169E1",
    "angry":    "#DC143C",
    "fear":     "#8B008B",
    "surprise": "#FF8C00",
    "disgust":  "#228B22",
}


# ── Image input normalization ───────────────────────────────────────────
def _coerce_image_bytes(image_input) -> bytes:
    """Accept bytes, bytearray, memoryview, numpy arrays, or PIL images."""
    if image_input is None:
        return b""

    if isinstance(image_input, (bytes, bytearray, memoryview)):
        return bytes(image_input)

    if NUMPY_AVAILABLE and isinstance(image_input, np.ndarray):
        if PILLOW_AVAILABLE:
            try:
                pil_image = Image.fromarray(image_input)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                return buffer.getvalue()
            except Exception:
                pass
        if CV2_AVAILABLE:
            try:
                success, encoded = cv2.imencode(".jpg", image_input)
                if success:
                    return encoded.tobytes()
            except Exception:
                pass
        return b""

    if PILLOW_AVAILABLE and hasattr(image_input, "save"):
        try:
            buffer = io.BytesIO()
            image_input.save(buffer, format="JPEG")
            return buffer.getvalue()
        except Exception:
            pass

    return b""


# ── Heuristic analysis helpers ────────────────────────────────────────────
def _pillow_analyze(image_bytes: bytes) -> dict:
    """
    Analyse emotion using Pillow by computing average channel statistics.
    Returns raw emotion probability dict (not yet normalised).
    """
    if not PILLOW_AVAILABLE:
        return _bytes_analyze(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((64, 64))  # small for speed
        pixels = list(img.getdata())

        total = len(pixels)
        r_avg = sum(p[0] for p in pixels) / total
        g_avg = sum(p[1] for p in pixels) / total
        b_avg = sum(p[2] for p in pixels) / total
        brightness = (r_avg + g_avg + b_avg) / 3.0  # 0-255

        # Ratios (0-1)
        r_ratio = r_avg / max(r_avg + g_avg + b_avg, 1)
        g_ratio = g_avg / max(r_avg + g_avg + b_avg, 1)
        b_ratio = b_avg / max(r_avg + g_avg + b_avg, 1)

        return _map_stats_to_emotions(brightness, r_ratio, g_ratio, b_ratio)
    except Exception:
        return _bytes_analyze(image_bytes)


def _bytes_analyze(image_bytes: bytes) -> dict:
    """
    Absolute fallback: derive emotion from raw byte statistics when
    no imaging library is available.
    """
    data = image_bytes[:4096] if len(image_bytes) > 4096 else image_bytes
    values = [b for b in data]
    brightness = sum(values) / max(len(values), 1) / 255.0 * 255  # 0-255
    # crude channel estimate from interleaved bytes
    r_avg = sum(values[0::3]) / max(len(values[0::3]), 1)
    g_avg = sum(values[1::3]) / max(len(values[1::3]), 1)
    b_avg = sum(values[2::3]) / max(len(values[2::3]), 1)
    total = max(r_avg + g_avg + b_avg, 1)
    return _map_stats_to_emotions(brightness, r_avg / total, g_avg / total, b_avg / total)


def _map_stats_to_emotions(brightness: float, r_ratio: float,
                            g_ratio: float, b_ratio: float) -> dict:
    """
    Map pixel statistics to raw emotion scores.

    Logic:
      - High brightness + high red ratio   → happy / surprise
      - Low brightness + high blue ratio   → sad / fear
      - High red ratio + low brightness    → angry
      - Balanced channels, medium bright   → neutral
      - High green, medium brightness      → slight disgust offset
    """
    scores = {e: 5.0 for e in EMOTIONS}  # base uniform prior

    norm_bright = brightness / 255.0  # 0-1

    # happy: bright warm image
    scores["happy"] += norm_bright * 25 + r_ratio * 20

    # surprise: very bright
    scores["surprise"] += norm_bright * 15 + (1 - abs(norm_bright - 0.8)) * 10

    # sad: dark + blue dominant
    scores["sad"] += (1 - norm_bright) * 25 + b_ratio * 15

    # fear: dark image
    scores["fear"] += (1 - norm_bright) * 20 + b_ratio * 10

    # angry: red dominant + medium-low brightness
    scores["angry"] += r_ratio * 25 + (1 - norm_bright) * 10

    # neutral: balanced channels, moderate brightness
    channel_balance = 1 - (abs(r_ratio - 0.33) + abs(g_ratio - 0.33) + abs(b_ratio - 0.33))
    scores["neutral"] += channel_balance * 30 + norm_bright * 10

    # disgust: greenish tones
    scores["disgust"] += g_ratio * 15

    # Clamp to ≥ 0
    for k in scores:
        scores[k] = max(0.0, scores[k])

    return scores


def _normalize_scores(raw: dict) -> dict:
    """Normalize raw emotion scores so they sum to 100."""
    total = sum(raw.values())
    if total == 0:
        equal = 100.0 / len(EMOTIONS)
        return {e: round(equal, 2) for e in EMOTIONS}
    return {e: round(raw.get(e, 0) / total * 100, 2) for e in EMOTIONS}


def _deepface_analyze(image_bytes: bytes) -> dict:
    """Run DeepFace analysis; returns normalized emotion dict or raises."""
    # Save bytes to temp file (DeepFace works best with file paths)
    import tempfile, os
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = DeepFace.analyze(
            img_path=tmp_path,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        emotion_raw = result.get("emotion", {})
        # DeepFace returns percentages already; just ensure all 7 are present
        normalized = {}
        for e in EMOTIONS:
            normalized[e] = round(float(emotion_raw.get(e, 0.0)), 2)
        # Re-normalise just in case DeepFace returns slightly different totals
        return _normalize_scores(normalized)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Public API ────────────────────────────────────────────────────────────
def analyze_image(image_bytes: bytes, filename: str = "image.jpg") -> dict:
    """
    Analyze emotion from image bytes or image-like inputs.

    Returns:
    {
      "emotions": {"happy": float, ...},  # all 7 emotions, sum ~100
      "dominant": str,
      "confidence": float,
      "engine": str,          # "deepface", "heuristic", or "error"
      "error": str | None,
    }
    """
    image_bytes = _coerce_image_bytes(image_bytes)

    if not image_bytes:
        return {
            "emotions": {e: round(100 / len(EMOTIONS), 2) for e in EMOTIONS},
            "dominant": "neutral",
            "confidence": 0.0,
            "engine": "error",
            "error": "Empty image bytes provided.",
        }

    # ── Try DeepFace first ────────────────────────────────────────────────
    if DEEPFACE_AVAILABLE:
        try:
            emotions = _deepface_analyze(image_bytes)
            dominant = max(emotions, key=emotions.get)
            return {
                "emotions": emotions,
                "dominant": dominant,
                "confidence": round(emotions[dominant], 2),
                "engine": "deepface",
                "error": None,
            }
        except Exception as exc:
            pass  # fall through to heuristic

    # ── Heuristic fallback ────────────────────────────────────────────────
    try:
        raw = _pillow_analyze(image_bytes)
        emotions = _normalize_scores(raw)
        dominant = max(emotions, key=emotions.get)
        return {
            "emotions": emotions,
            "dominant": dominant,
            "confidence": round(emotions[dominant], 2),
            "engine": "heuristic",
            "error": None,
        }
    except Exception as exc:
        # Absolute last resort: uniform distribution
        equal = round(100.0 / len(EMOTIONS), 2)
        emotions = {e: equal for e in EMOTIONS}
        return {
            "emotions": emotions,
            "dominant": "neutral",
            "confidence": equal,
            "engine": "error",
            "error": str(exc),
        }


def analyze_video_frames(
    video_bytes: bytes,
    filename: str = "video.mp4",
    max_frames: int = 10,
) -> dict:
    """
    Sample frames from video and aggregate emotion analysis.

    Returns same structure as analyze_image() plus:
    {
      "frame_count": int,
      "frame_results": [per-frame dict],
      "timeline": [{"frame": int, "dominant": str, "confidence": float}],
    }
    Fallback (no cv2): returns a heuristic result with frame_count=0.
    """
    base_result_template = {
        "emotions": {e: 0.0 for e in EMOTIONS},
        "dominant": "neutral",
        "confidence": 0.0,
        "engine": "heuristic",
        "error": None,
        "frame_count": 0,
        "frame_results": [],
        "timeline": [],
    }

    # ── cv2 path ──────────────────────────────────────────────────────────
    if CV2_AVAILABLE and video_bytes:
        import tempfile, os
        suffix = ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = max_frames

            step = max(1, total_frames // max_frames)
            frame_results = []
            timeline = []
            frame_idx = 0

            while cap.isOpened() and len(frame_results) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    success, buf = cv2.imencode(".jpg", frame)
                    if success:
                        img_bytes = buf.tobytes()
                        res = analyze_image(img_bytes, filename=f"frame_{frame_idx}.jpg")
                        res["frame_index"] = frame_idx
                        frame_results.append(res)
                        timeline.append({
                            "frame": frame_idx,
                            "dominant": res["dominant"],
                            "confidence": res["confidence"],
                        })
                frame_idx += 1

            cap.release()
            os.unlink(tmp_path)

            if not frame_results:
                raise ValueError("No frames extracted")

            # Aggregate by averaging emotion probabilities
            agg = {e: 0.0 for e in EMOTIONS}
            for fr in frame_results:
                for e in EMOTIONS:
                    agg[e] += fr["emotions"].get(e, 0.0)
            n = len(frame_results)
            agg = {e: round(agg[e] / n, 2) for e in EMOTIONS}
            dominant = max(agg, key=agg.get)
            engine = frame_results[0].get("engine", "heuristic") if frame_results else "heuristic"

            return {
                "emotions": agg,
                "dominant": dominant,
                "confidence": round(agg[dominant], 2),
                "engine": engine,
                "error": None,
                "frame_count": n,
                "frame_results": frame_results,
                "timeline": timeline,
            }

        except Exception as exc:
            # Fall through to heuristic
            pass

    # ── No cv2: heuristic on the raw bytes ───────────────────────────────
    try:
        raw = _pillow_analyze(video_bytes) if video_bytes else {e: 5.0 for e in EMOTIONS}
        emotions = _normalize_scores(raw)
        dominant = max(emotions, key=emotions.get)
        result = {
            "emotions": emotions,
            "dominant": dominant,
            "confidence": round(emotions[dominant], 2),
            "engine": "heuristic",
            "error": "cv2 not available; heuristic applied to raw bytes." if not CV2_AVAILABLE else None,
            "frame_count": 0,
            "frame_results": [],
            "timeline": [],
        }
        return result
    except Exception as exc:
        base_result_template["error"] = str(exc)
        return base_result_template


def calculate_behavioral_scores(emotion_result: dict) -> dict:
    """
    Convert emotion probabilities into behavioral scores.

    Returns:
    {
      "confidence_score": float,   # 0-100
      "engagement_score": float,   # 0-100
      "stability_score":  float,   # 0-100
      "behavioral_score": float,   # weighted overall
    }

    Formula:
      confidence = happy*0.6 + neutral*0.3 + surprise*0.1
      engagement  = (100 - sad - fear - angry)            clamped 0-100
      stability   = (neutral*0.5 + happy*0.3 + (100 - angry - fear)*0.2)  clamped 0-100
      behavioral  = 0.4*confidence + 0.3*engagement + 0.3*stability
    """
    emotions = emotion_result.get("emotions", {})

    happy    = emotions.get("happy",    0.0)
    neutral  = emotions.get("neutral",  0.0)
    sad      = emotions.get("sad",      0.0)
    angry    = emotions.get("angry",    0.0)
    fear     = emotions.get("fear",     0.0)
    surprise = emotions.get("surprise", 0.0)

    confidence_score = happy * 0.6 + neutral * 0.3 + surprise * 0.1
    engagement_score = max(0.0, min(100.0, 100 - sad - fear - angry))
    stability_score  = max(0.0, min(100.0,
        neutral * 0.5 + happy * 0.3 + (100 - angry - fear) * 0.2))
    behavioral_score = (0.4 * confidence_score
                        + 0.3 * engagement_score
                        + 0.3 * stability_score)

    return {
        "confidence_score": round(confidence_score, 2),
        "engagement_score": round(engagement_score, 2),
        "stability_score":  round(stability_score,  2),
        "behavioral_score": round(behavioral_score, 2),
    }


def get_emotion_color(emotion: str) -> str:
    """Return a hex color string for each emotion for charts."""
    return EMOTION_COLORS.get(emotion.lower(), "#AAAAAA")
