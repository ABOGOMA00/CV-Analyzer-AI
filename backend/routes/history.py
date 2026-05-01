import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database import get_db
from backend.models import Analysis
from backend.schemas import AnalysisHistoryItem

router = APIRouter()


@router.get("/", response_model=List[AnalysisHistoryItem])
def get_history(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    limit:   int           = Query(20,   ge=1, le=100, description="Max results to return"),
    skip:    int           = Query(0,    ge=0,         description="Pagination offset"),
    db:      Session       = Depends(get_db),
):
    """
    Returns analysis history ordered newest-first.
    Optionally filter by user_id and paginate with skip/limit.
    """
    query = db.query(Analysis)

    if user_id:
        query = query.filter(Analysis.user_id == user_id)

    analyses = (
        query
        .order_by(Analysis.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return analyses


@router.get("/{analysis_id}")
def get_analysis_detail(analysis_id: int, db: Session = Depends(get_db)):
    """Returns full detail for a single analysis, with JSON fields deserialised."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id":              analysis.id,
        "cv_filename":     analysis.cv_filename,
        "predicted_role":  analysis.predicted_role,
        "confidence":      analysis.confidence,
        "ats_score":       analysis.ats_score,
        "all_scores":      json.loads(analysis.all_scores) if analysis.all_scores else {},
        "tips":            json.loads(analysis.tips)       if analysis.tips       else {},
        "created_at":      analysis.created_at,
    }


@router.delete("/{analysis_id}")
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Permanently deletes an analysis record."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    db.delete(analysis)
    db.commit()

    return {"message": f"Analysis {analysis_id} deleted successfully"}
