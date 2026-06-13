"""
CV Analyzer AI -- Canonical Training Pipeline
=============================================
Best quick accuracy: TF-IDF (word+char) + MiniLM embeddings + LinearSVC.

Artifacts written to `saved_model/`:
- model.pkl (LinearSVC)
- label_encoder.pkl
- tfidf_word.pkl
- tfidf_char.pkl
- scaler.pkl (None; kept for backward compat)
"""

import os

# Determinism / stable results across runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")

CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "clean_resume_data.csv"
OUTPUT_DIR = "saved_model"
RANDOM_STATE = 42
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH") or next(
    (
        str(path)
        for path in (Path.home() / ".cache" / "huggingface" / "hub").glob(
            "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
        )
        if path.is_dir()
    ),
    EMBED_MODEL_NAME,
)

_PROTECTED_TERMS = sorted(
    [
        "c#", "c++", "r", "ai", "ml", "dl", "ui", "ux", "qa", "hr", "it", "bi",
        "go", "js", "ts", "ci", "cd", "db", "os", "vm", "gcp", "aws", "api",
        "sql", "css", "php", "ios", "rpa", "erp", "crm", "sap", "nlp", "cv",
        "dba", "cfo", "cto", "ceo", "vp", "pm", "ba", "qc", "ehr", "emr",
    ],
    key=len,
    reverse=True,
)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
    text = text.lower()

    protected_map = {}
    for term in _PROTECTED_TERMS:
        safe_key = f"PROT_{term.replace('#', 'SHARP').replace('+', 'PLUS')}__"
        if term in text:
            protected_map[safe_key] = term
            text = text.replace(term, f" {safe_key} ")

    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\b(?!PROT_)\w{1,2}\b", " ", text)

    for safe_key, original in protected_map.items():
        text = text.replace(safe_key, original)

    return re.sub(r"\s+", " ", text).strip()


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.atleast_2d(scores).astype(float)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_scores / denom


def main() -> int:
    print("\n" + "=" * 60)
    print("  CV ANALYZER AI -- CANONICAL TRAINING PIPELINE")
    print(f"  Strategy: TF-IDF(word+char) + {EMBED_MODEL_NAME} + LinearSVC")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH).dropna(subset=["Feature", "Category"]).copy()
    df["clean"] = df["Feature"].apply(clean_text)
    df = df[df["clean"].str.strip() != ""].copy()

    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    x = df["clean"].tolist()

    # Split data (keep original indices for later analysis)
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        df.index.to_numpy(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[+] Rows: {len(df)} | Classes: {len(le.classes_)}")
    print(f"[+] Train: {len(x_train)} | Test: {len(x_test)}")

    print("[*] Fitting TF-IDF (word)...")
    tfidf_word = TfidfVectorizer(
        max_features=40_000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        max_df=0.97,
        analyzer="word",
    )
    xw_train = tfidf_word.fit_transform(x_train)
    xw_test = tfidf_word.transform(x_test)

    print("[*] Fitting TF-IDF (char)...")
    tfidf_char = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(3, 6),
        sublinear_tf=True,
        min_df=1,
        max_df=0.97,
        analyzer="char_wb",
    )
    xc_train = tfidf_char.fit_transform(x_train)
    xc_test = tfidf_char.transform(x_test)

    x_train_feat = hstack([xw_train, xc_train])
    x_test_feat = hstack([xw_test, xc_test])
    print(f"[+] Feature shapes: train={x_train_feat.shape} test={x_test_feat.shape}")

    print("[*] Training LinearSVC...")
    model = LinearSVC(
        C=0.5,
        class_weight=None,
        random_state=RANDOM_STATE,
        max_iter=25_000,
    )
    model.fit(x_train_feat, y_train)

    y_pred = model.predict(x_test_feat)
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Holdout Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    joblib.dump(tfidf_word, os.path.join(OUTPUT_DIR, "tfidf_word.pkl"))
    joblib.dump(tfidf_char, os.path.join(OUTPUT_DIR, "tfidf_char.pkl"))
    joblib.dump(None, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    scores = model.decision_function(x_test_feat[:1])
    probs = _softmax_rows(scores)[0]
    print(f"[+] Example top confidence: {probs.max()*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
