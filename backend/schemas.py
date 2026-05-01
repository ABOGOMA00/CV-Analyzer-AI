from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


# ── User ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name:  str
    email: EmailStr


class UserResponse(BaseModel):
    id:         int
    name:       str
    email:      str
    created_at: datetime
    model_config = {"from_attributes": True}


# ── Sub-specialization ────────────────────────────────────────────────────────

class SubSpecScore(BaseModel):
    name:  str
    score: float


class SubSpecialization(BaseModel):
    top:    Optional[str]          = None
    scores: List[SubSpecScore]     = Field(default_factory=list)


class RelatedRole(BaseModel):
    role:    str
    display: str
    emoji:   str


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Full result returned after CV analysis."""
    id:               int
    cv_filename:      str
    cv_text:          Optional[str]              = None
    predicted_role:   str
    role_display:     Optional[str]              = None
    confidence:       float
    sector:           Optional[str]              = None
    sector_color:     Optional[str]              = None
    sector_icon:      Optional[str]              = None
    sub_specialization: Optional[SubSpecialization] = None
    career_level:     Optional[str]              = None
    related_roles:    Optional[List[RelatedRole]] = None
    ats_score:        Optional[float]            = None
    extracted_skills: Optional[List[str]]        = None
    missing_skills:   Optional[List[str]]        = None
    is_mismatch:      Optional[bool]             = None
    all_scores:       Optional[Dict[str, float]] = None
    tips:             Optional[Any]              = None
    created_at:       datetime

    model_config = {"from_attributes": True}


class AnalysisHistoryItem(BaseModel):
    """Lightweight history list item."""
    id:             int
    cv_filename:    str
    predicted_role: str
    confidence:     float
    ats_score:      Optional[float] = None
    sector:         Optional[str]   = None
    sector_color:   Optional[str]   = None
    created_at:     datetime

    model_config = {"from_attributes": True}

# ── Rewrite ───────────────────────────────────────────────────────────────────

class RewriteRequest(BaseModel):
    cv_text: str
    job_description: str

class RewriteResponse(BaseModel):
    rewritten_cv: str
    old_ats_score: Optional[float] = None
    new_ats_score: Optional[float] = None

class RewriteDownloadRequest(BaseModel):
    rewritten_cv: str
