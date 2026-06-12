from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base
import datetime

# Timezone-aware UTC helper — replaces the deprecated datetime.utcnow()
_utcnow = lambda: datetime.datetime.now(datetime.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    email      = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    cv_filename    = Column(String, nullable=False)
    # cv_text is capped at 5 000 chars to avoid bloating the DB with large CVs.
    # Full text is still available via the original uploaded file.
    cv_text        = Column(Text, nullable=True)
    predicted_role = Column(String, nullable=False)
    confidence     = Column(Float, nullable=False)
    ats_score      = Column(Float, nullable=True)
    all_scores     = Column(Text, nullable=True)
    tips           = Column(Text, nullable=True)
    created_at     = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="analyses")


class History(Base):
    """
    DEPRECATED — no application code writes to this table.
    Kept only so that existing databases (which already have the `history` table)
    don't break on startup.  Safe to drop via a future migration once you are
    sure the table is empty / no longer needed.
    """
    __tablename__ = "history"

    id             = Column(Integer, primary_key=True, index=True)
    cv_filename    = Column(String)
    predicted_role = Column(String)
    confidence     = Column(Float)
    created_at     = Column(DateTime(timezone=True), default=_utcnow)