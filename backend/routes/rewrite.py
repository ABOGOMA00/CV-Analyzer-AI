"""
Rewrite route — AI-powered CV rewriting with ATS score comparison.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
from backend.schemas import RewriteRequest, RewriteResponse, RewriteDownloadRequest
from backend.services.llm_service import generate_rewritten_cv

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    pass

router = APIRouter()


def _strip_alignment_block(text: str) -> str:
    """
    Remove ATS keyword list section before re-scoring to avoid artificial inflation.
    """
    marker = "\nATS Keyword Alignment:"
    idx = text.find(marker)
    return text[:idx].strip() if idx != -1 else text.strip()


def _extract_rewrite_body_for_scoring(text: str) -> str:
    """
    Score rewrite quality on substantive content (experience bullets),
    not on generated headings/keyword lists.
    """
    base = _strip_alignment_block(text)
    marker = "Experience Highlights:"
    idx = base.find(marker)
    if idx == -1:
        return base
    body = base[idx + len(marker):].strip()
    return body if body else base


@router.post("/", response_model=RewriteResponse)
async def rewrite_cv_endpoint(request: RewriteRequest):
    """
    Rewrite the provided CV to better match the target job description.
    Returns the rewritten text plus before/after ATS scores.
    """
    if not request.cv_text.strip():
        raise HTTPException(status_code=400, detail="CV text cannot be empty.")
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    try:
        from backend.services.ml_service import predict

        rewritten_text = generate_rewritten_cv(
            cv_text=request.cv_text,
            job_description=request.job_description,
        )
        rewritten_for_scoring = _extract_rewrite_body_for_scoring(rewritten_text)

        # 2. Calculate ATS scores locally (fast, no external API)
        old_ats = predict(request.cv_text, target_jd=request.job_description).get("ats_score")
        new_ats = predict(rewritten_for_scoring, target_jd=request.job_description).get("ats_score")

        return RewriteResponse(
            rewritten_cv=rewritten_text,
            old_ats_score=old_ats,
            new_ats_score=new_ats,
        )
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rewrite CV: {e}")

@router.post("/download")
async def download_rewritten_cv(request: RewriteDownloadRequest):
    """
    Generate a styled, ATS-compliant Word Document from the rewritten CV text.
    """
    if not request.rewritten_cv.strip():
        raise HTTPException(status_code=400, detail="CV text cannot be empty.")

    try:
        if 'Document' not in globals():
            raise HTTPException(status_code=500, detail="python-docx library is not installed.")

        doc = Document()
        
        # Setup basic ATS-friendly styling
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
        
        # Set margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        # Clean text
        text = request.rewritten_cv
        # Parse text into document
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic for headers (Markdown or uppercase)
            if line.startswith('# '):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line[2:].strip())
                run.bold = True
                run.font.size = Pt(16)
            elif line.startswith('## '):
                p = doc.add_paragraph()
                run = p.add_run(line[3:].strip())
                run.bold = True
                run.font.size = Pt(14)
            elif line.startswith('### '):
                p = doc.add_paragraph()
                run = p.add_run(line[4:].strip())
                run.bold = True
                run.font.size = Pt(12)
            elif line.isupper() and 3 < len(line) < 50:
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(12)
            elif line.startswith('- ') or line.startswith('* '):
                p = doc.add_paragraph(line[2:], style='List Bullet')
            else:
                # remove any residual markdown bold tags **
                clean_line = line.replace('**', '')
                doc.add_paragraph(clean_line)

        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        headers = {
            'Content-Disposition': 'attachment; filename="Optimized_CV.docx"'
        }
        
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {e}")
