import json
import logging
import re
from pathlib import PurePosixPath
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models import Analysis
from backend.schemas import AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter()
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = (".pdf", ".txt")


def _sanitize_filename(name: str | None) -> str:
    """Strip path components and null bytes from user-supplied filename."""
    if not name:
        return "untitled"
    # Remove null bytes and path traversal
    name = name.replace("\x00", "")
    name = PurePosixPath(name).name  # strip directory components
    return name[:255] if name else "untitled"


def _collapse_spaced_chars(text: str) -> str:
    """
    PDF extractors sometimes emit 'E x a m p l e' instead of 'Example'.
    Collapse those sequences so our boilerplate patterns can match.
    Only collapses runs of 2+ single chars separated by single spaces.
    """
    return re.sub(
        r'(?<![A-Za-z])([A-Za-z])(?: ([A-Za-z])){1,}(?![A-Za-z])',
        lambda m: m.group().replace(' ', ''),
        text,
    )


def _clean_cv_text(text: str) -> str:
    """
    Remove template-builder boilerplate that PDF extraction sometimes picks up.
    Common offenders: CV Genius, Novoresume, Zety, Resume.io — they embed
    promotional text ("Cover letter builder", "How to write a CV", etc.)
    at the end of the document.  We truncate at the first boilerplate marker.
    """


    # Normalise spaced-character PDF artifacts for pattern matching only
    normalised = _collapse_spaced_chars(text)

    # Ordered list of patterns that signal the start of boilerplate.
    # The CV content always precedes these sections.
    BOILERPLATE_PATTERNS = [
        r"(?i)\bhow\s+to\s+write\s+a\s+(cv|resume)\b",
        r"(?i)\bcover\s+letter\s+(builder|examples?|template|resources?)\b",
        r"(?i)\breading\s+our\s+articles\b",
        r"(?i)\bcv\s+(layout|examples?\s+by\s+industry|maker)\b",
        r"(?i)\bresume\s+(examples?|templates?|builder)\b",
        r"(?i)\bdownload\s+a\s+matching\s+cover\s+letter\b",
    ]

    # Watermarks to just remove completely from the text
    WATERMARKS = [
        r"(?i)\bexample\s+by\s+(cv\s*genius|novoresume|zety)\b",
        r"(?i)cv\s*genius",
    ]

    for pattern in BOILERPLATE_PATTERNS:
        m = re.search(pattern, normalised)
        if m and m.start() > 200:   # keep at least first 200 chars
            text = text[:m.start()].strip()
            # update normalised for subsequent checks
            normalised = normalised[:m.start()].strip()
            break   # truncate once and stop

    for w in WATERMARKS:
        # Remove watermarks from original text
        text = re.sub(w, "", text, flags=re.IGNORECASE)
        # Also try matching against the collapsed version to catch spaced-out watermarks
        collapsed = _collapse_spaced_chars(text)
        if collapsed != text:
            text = re.sub(w, "", collapsed, flags=re.IGNORECASE)

    return text.strip()


def _ocr_extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pdf2image and pytesseract (OCR)."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        logger.warning("OCR libraries (pytesseract/pdf2image) not installed.")
        return ""

    try:
        images = convert_from_bytes(content)
        texts = []
        for img in images:
            text = pytesseract.image_to_string(img)
            texts.append(text)
        return "\n".join(texts)
    except Exception as e:
        logger.warning(f"OCR extraction failed (make sure Tesseract-OCR is installed on the system): {e}")
        return ""


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF, with OCR fallback."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="PDF support requires PyMuPDF. Run: pip install PyMuPDF",
        )

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        
        cleaned = _clean_cv_text(text)
        if len(cleaned.strip()) < 50:
            logger.info("Extracted PDF text is empty or very short. Running OCR fallback...")
            ocr_text = _ocr_extract_pdf(content)
            if ocr_text.strip():
                return _clean_cv_text(ocr_text)
        return cleaned
    except Exception as e:
        logger.exception("PyMuPDF text extraction failed. Attempting OCR fallback...")
        try:
            ocr_text = _ocr_extract_pdf(content)
            if ocr_text.strip():
                return _clean_cv_text(ocr_text)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")



@router.post("/upload", response_model=AnalysisResponse)
def upload_and_analyze(
    file:      UploadFile    = File(...),
    target_jd: str           = Form(""),          # optional job description
    user_id:   Optional[int] = Form(None),        # optional user association
    db:        Session       = Depends(get_db),
):
    """
    Upload a CV (PDF or TXT), run AI analysis, persist the result, and
    return the full analysis including role prediction, ATS score, skills,
    and improvement tips.
    """
    # 1. Read file content
    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum supported size is 5 MB.",
        )

    # 2. Extract text based on file type
    filename = (file.filename or "").lower()
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or TXT file.",
        )
    if filename.endswith(".pdf"):
        cv_text = _extract_text_from_pdf(content)
    else:
        # TXT / plain-text fallback
        try:
            cv_text = content.decode("utf-8")
        except UnicodeDecodeError:
            cv_text = content.decode("latin-1")

    if not cv_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from CV — file appears to be empty.",
        )

    # 3. AI analysis
    try:
        from backend.services.ml_service import predict

        result = predict(cv_text, target_jd=target_jd)
    except Exception as e:
        logger.exception("AI analysis failed")
        raise HTTPException(status_code=500, detail="AI analysis failed. Please try again.")

    # 4. Persist to database
    try:
        analysis = Analysis(
            user_id        = user_id,
            cv_filename    = _sanitize_filename(file.filename),
            cv_text        = cv_text[:5000],
            predicted_role = result["predicted_role"],
            confidence     = result["confidence"],
            ats_score      = result.get("ats_score"),
            all_scores     = json.dumps(result["all_scores"]),
            tips           = json.dumps(result["tips"]),
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
    except Exception as e:
        db.rollback()
        logger.exception("Database error during analysis persistence")
        raise HTTPException(status_code=500, detail="Failed to save analysis results.")

    # 5. Build response — attach all enriched fields (not stored in DB, computed fresh)
    analysis.all_scores        = result["all_scores"]                    # type: ignore[assignment]
    analysis.tips              = result["tips"]                           # type: ignore[assignment]
    analysis.extracted_skills  = result.get("extracted_skills")          # type: ignore[attr-defined]
    analysis.missing_skills    = result.get("missing_skills")            # type: ignore[attr-defined]
    analysis.role_display      = result.get("role_display")              # type: ignore[attr-defined]
    analysis.sector            = result.get("sector")                    # type: ignore[attr-defined]
    analysis.sector_color      = result.get("sector_color")              # type: ignore[attr-defined]
    analysis.sector_icon       = result.get("sector_icon")               # type: ignore[attr-defined]
    analysis.sub_specialization = result.get("sub_specialization")       # type: ignore[attr-defined]
    analysis.career_level      = result.get("career_level")              # type: ignore[attr-defined]
    analysis.related_roles     = result.get("related_roles")             # type: ignore[attr-defined]
    analysis.is_mismatch       = result.get("is_mismatch")               # type: ignore[attr-defined]
    analysis.ats_breakdown     = result["tips"].get("ats_breakdown")     # type: ignore[attr-defined]
    analysis.matched_keywords  = result["tips"].get("matched_keywords")  # type: ignore[attr-defined]
    analysis.missing_keywords  = result["tips"].get("missing_keywords")  # type: ignore[attr-defined]
    analysis.ats_recommendations = result["tips"].get("ats_recommendations") # type: ignore[attr-defined]
    analysis.resume_strengths  = result["tips"].get("resume_strengths")  # type: ignore[attr-defined]
    analysis.resume_weaknesses = result["tips"].get("resume_weaknesses") # type: ignore[attr-defined]

    return analysis
