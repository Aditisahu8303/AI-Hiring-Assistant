"""
modules/report_generator.py
Generate CSV and PDF hiring reports.
reportlab is used for PDF; falls back gracefully if not installed.
"""

import csv
import io
import os
from datetime import datetime
from typing import Optional

OUTPUT_DIR = "outputs"

# ── ReportLab optional import ─────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────
def _ensure_output_dir() -> str:
    """Create outputs/ directory relative to CWD if it doesn't exist."""
    out_path = os.path.join(os.getcwd(), OUTPUT_DIR)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def _safe_get(obj: dict, *keys, default="N/A"):
    """Try multiple key names and return first non-None, non-empty value."""
    for key in keys:
        val = obj.get(key)
        if val is not None and str(val).strip() not in ("", "None", "nan"):
            return val
    return default


def _fmt(val, decimals: int = 1, default: str = "0") -> str:
    """Format a numeric value as a string."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def _build_candidate_row(candidate: dict) -> dict:
    """Normalise a candidate dict to a flat row for CSV/PDF output."""
    return {
        "rank":               _safe_get(candidate, "rank", default=""),
        "name":               _safe_get(candidate, "candidate_name", "name", "Name", default="Unknown"),
        "role":               _safe_get(candidate, "role", "job_title", "position", default="N/A"),
        "experience":         _fmt(_safe_get(candidate, "experience_years", "experience", default=0), 1, "0"),
        "resume_score":       _fmt(_safe_get(candidate, "resume_score", default=0)),
        "interview_score":    _fmt(_safe_get(candidate, "interview_score", default=0)),
        "voice_score":        _fmt(_safe_get(candidate, "voice_score", default=0)),
        "behavioral_score":   _fmt(_safe_get(candidate, "behavioral_score", default=0)),
        "fraud_score":        _fmt(_safe_get(candidate, "fraud_score", default=0)),
        "final_score":        _fmt(_safe_get(candidate, "final_score", default=0)),
        "recommendation":     _safe_get(candidate, "recommendation", default="N/A"),
        "status":             _safe_get(candidate, "status", "risk_level", default="N/A"),
        "emotion_dominant":   _safe_get(candidate, "emotion_dominant", "dominant_emotion", "dominant", default="N/A"),
        "fraud_flags":        _safe_get(candidate, "fraud_flags", default="None"),
    }


# ── CSV generators ─────────────────────────────────────────────────────────
def generate_hiring_csv(
    candidates: list,
    filename: str = "final_hiring_report.csv",
) -> bytes:
    """
    Convert list of candidate dicts to CSV bytes.
    Columns: rank, name, role, experience, resume_score, interview_score,
             voice_score, behavioral_score, fraud_score, final_score,
             recommendation, status, emotion_dominant, fraud_flags
    Also saves file to outputs/ directory.
    Returns csv bytes.
    """
    fieldnames = [
        "rank", "name", "role", "experience",
        "resume_score", "interview_score", "voice_score",
        "behavioral_score", "fraud_score", "final_score",
        "recommendation", "status", "emotion_dominant", "fraud_flags",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for candidate in candidates:
        row = _build_candidate_row(candidate)
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8")

    # Save to disk
    try:
        out_dir = _ensure_output_dir()
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "wb") as f:
            f.write(csv_bytes)
    except Exception:
        pass  # Don't raise — still return the bytes

    return csv_bytes


def generate_summary_csv(
    exec_summary: dict,
    filename: str = "hiring_summary.csv",
) -> bytes:
    """
    Convert executive summary dict to a key-value CSV.
    Returns csv bytes.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Metric", "Value"])

    # Scalar values
    scalar_keys = [
        "total", "strong_hire", "hire", "consider", "reject",
        "avg_final_score", "hiring_success_rate", "avg_experience", "quality_index",
    ]
    for key in scalar_keys:
        if key in exec_summary:
            writer.writerow([key.replace("_", " ").title(), exec_summary[key]])

    # Top candidate
    top = exec_summary.get("top_candidate", {})
    if top:
        top_name  = _safe_get(top, "candidate_name", "name", default="Unknown")
        top_score = _fmt(_safe_get(top, "final_score", default=0))
        writer.writerow(["Top Candidate", f"{top_name} ({top_score})"])

    # Top skills
    top_skills = exec_summary.get("top_skills", [])
    if top_skills:
        skills_str = ", ".join(f"{s} ({c})" for s, c in top_skills[:5])
        writer.writerow(["Top Skills", skills_str])

    # Role performance
    role_perf = exec_summary.get("role_performance", {})
    for role, avg_score in role_perf.items():
        writer.writerow([f"Avg Score — {role}", f"{avg_score:.1f}"])

    # Most common fraud flags
    flags = exec_summary.get("most_common_flags", [])
    if flags:
        flags_str = "; ".join(f"{f} ({c})" for f, c in flags)
        writer.writerow(["Common Fraud Flags", flags_str])

    csv_bytes = output.getvalue().encode("utf-8")

    try:
        out_dir = _ensure_output_dir()
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "wb") as f:
            f.write(csv_bytes)
    except Exception:
        pass

    return csv_bytes


# ── PDF generator ──────────────────────────────────────────────────────────
def generate_hiring_pdf(
    candidates: list,
    exec_summary: dict,
    title: str = "AI Hiring Assistant \u2014 Final Report",
) -> Optional[bytes]:
    """
    Generate PDF using reportlab.
    Includes:
      - Title page with summary stats
      - Top 10 candidates table
      - Individual candidate scorecard for top 5
      - Recommendations section
    Returns PDF bytes or None if reportlab not available.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        spaceAfter=12,
        textColor=colors.HexColor("#1A3A5C"),
        alignment=TA_CENTER,
    )
    style_h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1A3A5C"),
        spaceBefore=14,
        spaceAfter=6,
    )
    style_h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#2C6E9C"),
        spaceBefore=10,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=4,
    )
    style_caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    story = []

    # ── Title page ─────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(title, style_title))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        style_caption,
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A3A5C")))
    story.append(Spacer(1, 0.5 * cm))

    # Summary stats table on title page
    total           = exec_summary.get("total", 0)
    strong_hire     = exec_summary.get("strong_hire", 0)
    hire            = exec_summary.get("hire", 0)
    consider        = exec_summary.get("consider", 0)
    reject          = exec_summary.get("reject", 0)
    avg_score       = exec_summary.get("avg_final_score", 0)
    success_rate    = exec_summary.get("hiring_success_rate", 0)
    quality_index   = exec_summary.get("quality_index", 0)

    summary_data = [
        ["Metric", "Value"],
        ["Total Candidates Evaluated", str(total)],
        ["Strong Hire",                str(strong_hire)],
        ["Hire",                       str(hire)],
        ["Consider",                   str(consider)],
        ["Reject",                     str(reject)],
        ["Average Final Score",        f"{avg_score:.1f} / 100"],
        ["Hiring Success Rate",        f"{success_rate:.1f}%"],
        ["Quality Index",              f"{quality_index:.2f}"],
    ]

    summary_table = Table(summary_data, colWidths=[10 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1A3A5C")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#EBF5FB"), colors.white]),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # ── Top 10 candidates table ────────────────────────────────────────
    story.append(Paragraph("Top 10 Candidates", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C6E9C")))
    story.append(Spacer(1, 0.3 * cm))

    # Sort candidates by final_score
    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("final_score", 0)),
        reverse=True,
    )[:10]

    top10_header = ["#", "Name", "Role", "Final\nScore", "Recommendation", "Fraud\nScore"]
    top10_data   = [top10_header]
    for i, c in enumerate(sorted_candidates, start=1):
        row = _build_candidate_row(c)
        top10_data.append([
            str(i),
            row["name"][:25],
            row["role"][:20],
            row["final_score"],
            row["recommendation"],
            row["fraud_score"],
        ])

    col_widths_top10 = [1 * cm, 5.5 * cm, 4.5 * cm, 2 * cm, 3.5 * cm, 2 * cm]
    top10_table = Table(top10_data, colWidths=col_widths_top10, repeatRows=1)
    top10_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2C6E9C")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#EBF5FB"), colors.white]),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",         (3, 0), (3, -1),  "CENTER"),
        ("ALIGN",         (5, 0), (5, -1),  "CENTER"),
    ]))
    story.append(top10_table)
    story.append(PageBreak())

    # ── Individual scorecards for top 5 ───────────────────────────────
    story.append(Paragraph("Individual Candidate Scorecards — Top 5", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C6E9C")))
    story.append(Spacer(1, 0.3 * cm))

    top5 = sorted_candidates[:5]
    for rank, c in enumerate(top5, start=1):
        row = _build_candidate_row(c)

        card_elements = []
        card_elements.append(
            Paragraph(f"#{rank} — {row['name']} | {row['role']}", style_h2)
        )

        score_data = [
            ["Component",           "Score"],
            ["Resume Score",         row["resume_score"]],
            ["Interview Score",      row["interview_score"]],
            ["Voice Score",          row["voice_score"]],
            ["Behavioral Score",     row["behavioral_score"]],
            ["Fraud Risk Score",     row["fraud_score"]],
            ["", ""],
            ["FINAL SCORE",          row["final_score"]],
            ["Recommendation",       row["recommendation"]],
            ["Dominant Emotion",     row["emotion_dominant"]],
            ["Fraud Flags",          str(row["fraud_flags"])[:60]],
        ]

        card_table = Table(score_data, colWidths=[7 * cm, 9 * cm])
        card_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),   colors.HexColor("#2C6E9C")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.white),
            ("FONTSIZE",      (0, 0), (-1, 0),   9),
            ("FONTSIZE",      (0, 1), (-1, -1),  8),
            ("BACKGROUND",    (0, 7), (-1, 7),   colors.HexColor("#EBF5FB")),
            ("FONTNAME",      (0, 7), (-1, 7),   "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, 6),
             [colors.white, colors.HexColor("#F8F9FA")]),
            ("ROWBACKGROUNDS",(0, 8), (-1, -1),
             [colors.white, colors.HexColor("#F8F9FA")]),
            ("BOX",           (0, 0), (-1, -1),  0.5, colors.grey),
            ("INNERGRID",     (0, 0), (-1, -1),  0.3, colors.lightgrey),
            ("LEFTPADDING",   (0, 0), (-1, -1),  6),
            ("TOPPADDING",    (0, 0), (-1, -1),  3),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  3),
        ]))
        card_elements.append(card_table)
        card_elements.append(Spacer(1, 0.4 * cm))

        # Reasoning
        reasoning = c.get("reasoning", "")
        if reasoning:
            card_elements.append(
                Paragraph(f"<i>{reasoning}</i>", style_body)
            )
        card_elements.append(Spacer(1, 0.3 * cm))
        card_elements.append(
            HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey)
        )
        card_elements.append(Spacer(1, 0.2 * cm))

        story.extend(card_elements)

    story.append(PageBreak())

    # ── Recommendations section ────────────────────────────────────────
    story.append(Paragraph("Hiring Recommendations", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C6E9C")))
    story.append(Spacer(1, 0.3 * cm))

    rec_groups = {"Strong Hire": [], "Hire": [], "Consider": [], "Reject": []}
    for c in candidates:
        rec = c.get("recommendation", "Reject")
        if rec not in rec_groups:
            rec = "Reject"
        rec_groups[rec].append(c)

    rec_colors = {
        "Strong Hire": "#27AE60",
        "Hire":        "#2980B9",
        "Consider":    "#F39C12",
        "Reject":      "#E74C3C",
    }

    for rec_label, rec_candidates in rec_groups.items():
        if not rec_candidates:
            continue
        hex_color = rec_colors[rec_label]
        story.append(
            Paragraph(
                f'<font color="{hex_color}"><b>{rec_label}</b></font> '
                f'({len(rec_candidates)} candidate{"s" if len(rec_candidates) != 1 else ""})',
                style_h2,
            )
        )
        names = []
        for c in sorted(rec_candidates, key=lambda x: float(x.get("final_score", 0)), reverse=True):
            cname = _safe_get(c, "candidate_name", "name", default="Unknown")
            fscore = _fmt(c.get("final_score", 0))
            names.append(f"• {cname} (Score: {fscore})")
        story.append(Paragraph("<br/>".join(names[:20]), style_body))
        story.append(Spacer(1, 0.3 * cm))

    # Top skills section
    top_skills = exec_summary.get("top_skills", [])
    if top_skills:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Most Common Skills in Applicant Pool", style_h2))
        skills_str = "  •  ".join(f"{s} ({c})" for s, c in top_skills[:10])
        story.append(Paragraph(skills_str, style_body))

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()

    # Save to disk
    try:
        out_dir = _ensure_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(out_dir, f"hiring_report_{ts}.pdf")
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
    except Exception:
        pass

    return pdf_bytes


# ── Plain-text 360° profile ────────────────────────────────────────────────
def generate_candidate_profile_txt(candidate: dict) -> str:
    """Generate plain text 360° profile for one candidate."""
    row = _build_candidate_row(candidate)

    # Component score bar (text-based)
    def score_bar(score_str: str, width: int = 20) -> str:
        try:
            val = float(score_str)
        except ValueError:
            val = 0.0
        filled = int(val / 100 * width)
        return "[" + "█" * filled + "░" * (width - filled) + f"]  {val:.1f}/100"

    # Fraud flags list
    flags_raw = str(row.get("fraud_flags", "None"))
    if flags_raw.lower() in ("none", "n/a", ""):
        flags_display = "  None detected."
    else:
        flags_display = "\n".join(f"  ⚠  {f.strip()}" for f in flags_raw.split(";") if f.strip())

    # Strengths from evaluation (if available)
    strengths = candidate.get("top_strengths", candidate.get("strengths", []))
    improvements = candidate.get("top_improvements", candidate.get("improvements", []))
    strengths_txt    = ("\n".join(f"  ✓  {s}" for s in strengths)
                        if strengths else "  No data available.")
    improvements_txt = ("\n".join(f"  →  {i}" for i in improvements)
                        if improvements else "  No data available.")

    # Reasoning / recommendation note
    reasoning = candidate.get("reasoning", "")

    sep = "=" * 60
    thin = "-" * 60

    profile = f"""{sep}
AI HIRING ASSISTANT — 360° CANDIDATE PROFILE
{sep}
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}
{thin}
CANDIDATE OVERVIEW
{thin}
  Name           : {row['name']}
  Role Applied   : {row['role']}
  Experience     : {row['experience']} years
  Recommendation : {row['recommendation']}
  Rank           : #{row['rank']}

{thin}
SCORE BREAKDOWN
{thin}
  Resume Score      : {score_bar(row['resume_score'])}
  Interview Score   : {score_bar(row['interview_score'])}
  Voice Score       : {score_bar(row['voice_score'])}
  Behavioral Score  : {score_bar(row['behavioral_score'])}
  Fraud Risk Score  : {score_bar(row['fraud_score'])}
  ─────────────────────────────────────────────────────
  FINAL SCORE       : {score_bar(row['final_score'])}

{thin}
BEHAVIORAL / EMOTION ANALYSIS
{thin}
  Dominant Emotion  : {row['emotion_dominant']}
  Status            : {row['status']}

{thin}
FRAUD RISK ASSESSMENT
{thin}
{flags_display}

{thin}
KEY STRENGTHS
{thin}
{strengths_txt}

{thin}
AREAS FOR IMPROVEMENT
{thin}
{improvements_txt}
"""

    if reasoning:
        profile += f"""
{thin}
AI DECISION RATIONALE
{thin}
  {reasoning}
"""

    profile += f"\n{sep}\n"
    return profile
