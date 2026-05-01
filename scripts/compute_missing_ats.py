#!/usr/bin/env python
"""Recompute ATS scores for Analysis records lacking them.

Usage:
    python scripts/compute_missing_ats.py --job-desc data/sample/sample_job.txt

The script loads all Analysis rows where ats_score is NULL, extracts the CV text,
computes a cosine-similarity ATS score against the provided job description using
the SentenceTransformer model defined in backend/services/ml_service.py, and
updates the database.
"""

import argparse
import json
import os
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Import project modules (adjust PYTHONPATH if needed)
import sys
# Ensure the project root (containing the 'backend' package) is on PYTHONPATH
# The script resides in <project_root>/scripts, so two levels up is the root.
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root

from backend.database import DATABASE_URL, Base
from backend.models import Analysis
from backend.services.ml_service import compute_ats_score


def get_job_description(path: str) -> str:
    """Read the job description file as plain text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute missing ATS scores.")
    parser.add_argument(
        "--job-desc",
        required=True,
        help="Path to a plain‑text job description to compare against.",
    )
    args = parser.parse_args()

    jd_text = get_job_description(args.job_desc)

    # Set up DB session
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        stmt = select(Analysis).where(Analysis.ats_score.is_(None))
        rows = session.execute(stmt).scalars().all()
        if not rows:
            print("No analyses with missing ATS score found.")
            return
        print(f"Found {len(rows)} records to update.")
        for analysis in rows:
            cv_text = analysis.cv_text or ""
            ats = compute_ats_score(cv_text, jd_text)
            analysis.ats_score = ats if ats >= 0 else None
            session.add(analysis)
        session.commit()
        print("ATS scores recomputed and saved.")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
