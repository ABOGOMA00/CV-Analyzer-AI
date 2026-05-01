import os
import json
from datetime import date
from pathlib import Path

# Make evaluation deterministic across runs by forcing single-threaded math.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import joblib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from train_model import clean_text
import os as _os
_APPLY_POSTPROCESS = _os.getenv("APPLY_POSTPROCESS", "0") == "1"
if _APPLY_POSTPROCESS:
    from backend.services.postprocess import adjust_predicted_role


RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_CONFUSIONS = 15
EXAMPLES_PER_CONFUSION = 3
MAX_SNIPPET_CHARS = 260
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


def _snippet(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> int:
    root = Path(__file__).resolve().parent

    model_path = root / "saved_model" / "model.pkl"
    encoder_path = root / "saved_model" / "label_encoder.pkl"
    data_path = root / "clean_resume_data.csv"

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not encoder_path.exists():
        raise SystemExit(f"Missing encoder: {encoder_path}")
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    tfidf_word_path = root / "saved_model" / "tfidf_word.pkl"
    tfidf_char_path = root / "saved_model" / "tfidf_char.pkl"
    has_hybrid = tfidf_word_path.exists() and tfidf_char_path.exists()

    tfidf_word = joblib.load(tfidf_word_path) if tfidf_word_path.exists() else None
    tfidf_char = joblib.load(tfidf_char_path) if tfidf_char_path.exists() else None

    df = pd.read_csv(data_path).dropna(subset=["Feature", "Category"]).copy()
    df["clean_text"] = df["Feature"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""].copy()

    y = encoder.transform(df["Category"])
    x = df["clean_text"].tolist()

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if has_hybrid:
        from sentence_transformers import SentenceTransformer

        embed = SentenceTransformer(EMBED_MODEL_PATH)
        xw = tfidf_word.transform(x_test)
        xc = tfidf_char.transform(x_test)
        emb = embed.encode(x_test, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        xt = hstack([xw, xc, csr_matrix(emb)])
        y_pred = model.predict(xt)
    else:
        y_pred = model.predict(x_test)

    def _softmax_rows(scores: np.ndarray) -> np.ndarray:
        scores = np.atleast_2d(scores).astype(float)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        denom = exp_scores.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return exp_scores / denom

    if _APPLY_POSTPROCESS:
        # Apply the same post-processing rules used by the API.
        for i in range(len(y_pred)):
            if has_hybrid:
                # Rebuild a single-row feature for probability estimation.
                xw1 = tfidf_word.transform([x_test[i]])
                xc1 = tfidf_char.transform([x_test[i]])
                emb1 = embed.encode([x_test[i]], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
                x1 = hstack([xw1, xc1, csr_matrix(emb1)])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(x1)[0]
                else:
                    probs = _softmax_rows(model.decision_function(x1))[0]
            else:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([x_test[i]])[0]
                else:
                    probs = _softmax_rows(model.decision_function([x_test[i]]))[0]
            all_scores = {encoder.classes_[j]: round(float(probs[j] * 100), 2) for j in range(len(encoder.classes_))}
            role = encoder.inverse_transform([y_pred[i]])[0]
            adjusted_role = adjust_predicted_role(
                cv_text=str(df.loc[idx_test[i], "Feature"]),
                predicted_role=role,
                all_scores=all_scores,
            )
            if adjusted_role != role:
                y_pred[i] = encoder.transform([adjusted_role])[0]

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    labels = list(range(len(encoder.classes_)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Collect confusion pairs (true -> pred) excluding diagonal.
    confusions = []
    for true_i in labels:
        for pred_i in labels:
            if true_i == pred_i:
                continue
            count = int(cm[true_i, pred_i])
            if count:
                confusions.append((count, true_i, pred_i))
    confusions.sort(reverse=True)
    top_confusions = confusions[:TOP_K_CONFUSIONS]

    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"error_analysis_{date.today().isoformat()}.md"

    # Build example index for quick retrieval
    test_rows = df.loc[idx_test].copy()
    test_rows["y_true"] = y_test
    test_rows["y_pred"] = y_pred

    lines = []
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append(f"- Date: `{date.today().isoformat()}`")
    lines.append(f"- Model: `{model_path.as_posix()}`")
    lines.append(f"- Dataset: `{data_path.as_posix()}` (rows used: `{len(df)}`)")
    lines.append(f"- Holdout split: test_size=`{TEST_SIZE}`, random_state=`{RANDOM_STATE}`")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append(f"- Accuracy: `{acc*100:.2f}%`")
    lines.append(f"- Macro F1: `{macro_f1:.3f}`")
    lines.append(f"- Weighted F1: `{weighted_f1:.3f}`")
    lines.append("")
    lines.append("## Top Confusions (True -> Predicted)")
    lines.append("")

    for rank, (count, true_i, pred_i) in enumerate(top_confusions, start=1):
        true_label = encoder.classes_[true_i]
        pred_label = encoder.classes_[pred_i]
        true_support = int(cm[true_i, :].sum())
        pct = (count / true_support * 100.0) if true_support else 0.0
        lines.append(f"{rank}. `{true_label}` -> `{pred_label}`: `{count}` ({pct:.1f}% of `{true_label}` test samples)")

        # Add a few examples
        subset = test_rows[(test_rows["y_true"] == true_i) & (test_rows["y_pred"] == pred_i)].head(EXAMPLES_PER_CONFUSION)
        if not subset.empty:
            for _, row in subset.iterrows():
                lines.append(f"   - {json.dumps(_snippet(row['Feature']), ensure_ascii=False)}")

    lines.append("")
    lines.append("## Full Classification Report")
    lines.append("")
    lines.append("```")
    lines.append(classification_report(y_test, y_pred, target_names=encoder.classes_, digits=3))
    lines.append("```")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[+] Wrote report: {report_path}")
    print(f"[+] Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.3f} | Weighted-F1: {weighted_f1:.3f}")
    if top_confusions:
        print("[+] Top confusions:")
        for count, true_i, pred_i in top_confusions[:8]:
            print(f"  - {encoder.classes_[true_i]} -> {encoder.classes_[pred_i]}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
