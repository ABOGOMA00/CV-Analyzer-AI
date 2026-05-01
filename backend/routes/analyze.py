import json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models import Analysis
from backend.schemas import AnalysisResponse

router = APIRouter()
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = (".pdf", ".txt")


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
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
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")


@router.post("/upload", response_model=AnalysisResponse)
async def upload_and_analyze(
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
    content = await file.read()
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
        raise HTTPException(status_code=500, detail=f"AI Analysis Error: {e}")

    # 4. Persist to database
    try:
        analysis = Analysis(
            user_id        = user_id,
            cv_filename    = file.filename,
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
        raise HTTPException(status_code=500, detail=f"Database Error: {e}")

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

    return analysis
