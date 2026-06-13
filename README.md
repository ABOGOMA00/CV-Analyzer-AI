# CV Analyzer AI

CV Analyzer AI is a local FastAPI + JavaScript project for resume classification, ATS matching, skill extraction, and analysis history tracking.

## What It Does

- Predicts one of 24 job categories from a CV
- Scores the CV against an optional job description
- Extracts skills and highlights missing skills
- Stores analysis history in SQLite
- Shows role, sector, related roles, and improvement tips in the UI

## Tech Stack

- Backend: FastAPI, SQLAlchemy, PyMuPDF, spaCy
- ML: scikit-learn + SentenceTransformers
- Frontend: HTML, CSS, JavaScript
- Storage: SQLite

## Current Model

- Training entrypoint: [train_model.py](./train_model.py)
- Features: TF-IDF word + TF-IDF char + MiniLM embeddings
- Classifier: LinearSVC
- Verified holdout accuracy on the current dataset: `98.90%`
- Training data: [data/raw/clean_resume_data.csv](./data/raw/clean_resume_data.csv)

## Project Layout

```text
backend/
Frontend/
Sample_CVs/
saved_model/
reports/
train_model.py
clean_resume_data.csv
README.md
```

## Setup

1. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Train or refresh the model:

```bash
python train_model.py
```

3. Start the API on Windows:

```bash
run_server.cmd
```

Or start it manually:

```bash
uvicorn backend.main:app --reload --port 8000
```

4. Open the UI:

```text
http://localhost:8000/app
```

## Notes

- The database file is stored at the project root as `cv_analyzer.db`.
- Trained artifacts are stored in `saved_model/`.
- The app automatically loads the local MiniLM cache if it exists.
- `EMBED_MODEL_PATH` can be set manually if you want to point to a custom local SentenceTransformer cache.

## Optional Environment Variables

- `ALLOWED_ORIGINS`: comma-separated CORS origins, default `*`
- `DATABASE_URL`: override the SQLite database path if needed
- `EMBED_MODEL_PATH`: custom local path to the MiniLM SentenceTransformer snapshot

## Sample Files

The `Sample_CVs/` folder contains a few ready-to-test resumes you can use during the demo.
