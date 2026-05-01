"""
prepare_resume_data.py
======================
Rebuilds data/processed/merged_resume_data.csv from scratch by combining:
  1. data/raw/resume_data.csv          (new dataset uploaded by user)
  2. data/raw/clean_resume_data.csv    (original base)
  3. data/raw/UpdatedResumeDataSet.csv (first extension)

Run from the project root:
    python scripts/prepare_resume_data.py
"""

import re
import sys
import ast
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
SRC_NEW     = ROOT / "data" / "raw" / "resume_data.csv"
SRC_CLEAN   = ROOT / "data" / "raw" / "clean_resume_data.csv"
SRC_UPDATED = ROOT / "data" / "raw" / "UpdatedResumeDataSet.csv"
DEST        = ROOT / "data" / "processed" / "merged_resume_data.csv"

# ── Job-position → Category mapping ───────────────────────────────────────────
# EXACT titles (lowercase) found in resume_data.csv → mapped precisely
JOB_MAP: dict[str, str] = {

    # ── Exact titles from resume_data.csv (28 unique) ─────────────────────────
    "ai engineer":                                                              "INFORMATION-TECHNOLOGY",
    "asst. manager/ manger (administrative)":                                   "HR",
    "business development executive":                                           "BUSINESS-DEVELOPMENT",
    "civil engineer":                                                           "ENGINEERING",
    "data engineer":                                                            "INFORMATION-TECHNOLOGY",
    "data science engineer":                                                    "INFORMATION-TECHNOLOGY",
    "database administrator (dba)":                                             "INFORMATION-TECHNOLOGY",
    "devops engineer":                                                          "INFORMATION-TECHNOLOGY",
    "executive - vat":                                                          "FINANCE",
    "executive/ senior executive- trade marketing, hygiene products":           "DIGITAL-MEDIA",
    "executive/ sr. executive -it":                                             "INFORMATION-TECHNOLOGY",
    "full stack developer (python,react js)":                                   "INFORMATION-TECHNOLOGY",
    "hr officer":                                                               "HR",
    "head of internal control & compliance (icc) - sevp/dmd":                  "FINANCE",
    "intern (generative ai engineering - 2d/3d image generation)":              "INFORMATION-TECHNOLOGY",
    "machine learning (ml) engineer":                                           "INFORMATION-TECHNOLOGY",
    "management trainee - mechanical":                                          "ENGINEERING",
    "manager- human resource management (hrm)":                                 "HR",
    "marketing officer":                                                        "DIGITAL-MEDIA",
    "mechanical designer":                                                      "DESIGNER",
    "mechanical engineer":                                                      "ENGINEERING",
    "network support engineer":                                                 "INFORMATION-TECHNOLOGY",
    "project coordinator (civil)":                                              "ENGINEERING",
    "senior software engineer":                                                 "INFORMATION-TECHNOLOGY",
    "senior ios engineer":                                                      "INFORMATION-TECHNOLOGY",
    "site engineer":                                                            "ENGINEERING",
    "sr.officer / executive - internal audit":                                  "FINANCE",
    "system administrator (operation & maintenance of server, storage & service desk system)": "INFORMATION-TECHNOLOGY",

    # ── Broad fuzzy fallbacks (for future data / other CSVs) ──────────────────

    # IT / Software
    "software engineer":               "INFORMATION-TECHNOLOGY",
    "software developer":              "INFORMATION-TECHNOLOGY",
    "python developer":                "INFORMATION-TECHNOLOGY",
    "java developer":                  "INFORMATION-TECHNOLOGY",
    "web developer":                   "INFORMATION-TECHNOLOGY",
    "frontend developer":              "INFORMATION-TECHNOLOGY",
    "backend developer":               "INFORMATION-TECHNOLOGY",
    "full stack developer":            "INFORMATION-TECHNOLOGY",
    "fullstack developer":             "INFORMATION-TECHNOLOGY",
    "mobile developer":                "INFORMATION-TECHNOLOGY",
    "android developer":               "INFORMATION-TECHNOLOGY",
    "ios developer":                   "INFORMATION-TECHNOLOGY",
    "ios engineer":                    "INFORMATION-TECHNOLOGY",
    "data scientist":                  "INFORMATION-TECHNOLOGY",
    "data analyst":                    "INFORMATION-TECHNOLOGY",
    "machine learning":                "INFORMATION-TECHNOLOGY",
    "ml engineer":                     "INFORMATION-TECHNOLOGY",
    "generative ai":                   "INFORMATION-TECHNOLOGY",
    "cloud engineer":                  "INFORMATION-TECHNOLOGY",
    "database administrator":          "INFORMATION-TECHNOLOGY",
    "system administrator":            "INFORMATION-TECHNOLOGY",
    "network engineer":                "INFORMATION-TECHNOLOGY",
    "cybersecurity analyst":           "INFORMATION-TECHNOLOGY",
    "security engineer":               "INFORMATION-TECHNOLOGY",
    "qa engineer":                     "INFORMATION-TECHNOLOGY",
    "testing engineer":                "INFORMATION-TECHNOLOGY",
    "etl developer":                   "INFORMATION-TECHNOLOGY",
    "hadoop developer":                "INFORMATION-TECHNOLOGY",
    "big data engineer":               "INFORMATION-TECHNOLOGY",
    "blockchain developer":            "INFORMATION-TECHNOLOGY",
    "dotnet developer":                "INFORMATION-TECHNOLOGY",
    "sap developer":                   "INFORMATION-TECHNOLOGY",
    "it executive":                    "INFORMATION-TECHNOLOGY",
    "it officer":                      "INFORMATION-TECHNOLOGY",

    # Engineering
    "electrical engineer":             "ENGINEERING",
    "chemical engineer":               "ENGINEERING",
    "industrial engineer":             "ENGINEERING",
    "structural engineer":             "ENGINEERING",
    "automation engineer":             "ENGINEERING",

    # Designer
    "ui/ux designer":                  "DESIGNER",
    "graphic designer":                "DESIGNER",
    "product designer":                "DESIGNER",
    "web designer":                    "DESIGNER",
    "visual designer":                 "DESIGNER",

    # Digital Media / Marketing
    "seo specialist":                  "DIGITAL-MEDIA",
    "digital marketing":               "DIGITAL-MEDIA",
    "content creator":                 "DIGITAL-MEDIA",
    "social media manager":            "DIGITAL-MEDIA",
    "marketing manager":               "DIGITAL-MEDIA",
    "trade marketing":                 "DIGITAL-MEDIA",

    # Finance / Audit
    "financial analyst":               "FINANCE",
    "investment banker":               "FINANCE",
    "risk analyst":                    "FINANCE",
    "portfolio manager":               "FINANCE",
    "finance manager":                 "FINANCE",
    "internal audit":                  "FINANCE",
    "internal control":                "FINANCE",
    "vat":                             "FINANCE",

    # Banking
    "retail banker":                   "BANKING",
    "credit analyst":                  "BANKING",
    "compliance officer":              "BANKING",
    "bank teller":                     "BANKING",
    "banking":                         "BANKING",

    # Sales / Business Development
    "sales manager":                   "SALES",
    "account executive":               "SALES",
    "sales representative":            "SALES",
    "sales":                           "SALES",
    "business development":            "BUSINESS-DEVELOPMENT",

    # HR / Admin
    "hr manager":                      "HR",
    "hr officer":                      "HR",
    "human resource management":       "HR",
    "human resources":                 "HR",
    "talent acquisition":              "HR",
    "recruiter":                       "HR",
    "hr business partner":             "HR",
    "administrative":                  "HR",

    # Teacher / Trainer
    "teacher":                         "TEACHER",
    "lecturer":                        "TEACHER",
    "professor":                       "TEACHER",
    "instructor":                      "TEACHER",
    "educator":                        "TEACHER",
    "trainer":                         "TEACHER",

    # Healthcare
    "doctor":                          "HEALTHCARE",
    "physician":                       "HEALTHCARE",
    "nurse":                           "HEALTHCARE",
    "pharmacist":                      "HEALTHCARE",
    "medical researcher":              "HEALTHCARE",
    "healthcare":                      "HEALTHCARE",

    # Accountant
    "accountant":                      "ACCOUNTANT",
    "auditor":                         "ACCOUNTANT",
    "tax accountant":                  "ACCOUNTANT",

    # Consultant
    "consultant":                      "CONSULTANT",
    "management consultant":           "CONSULTANT",

    # Chef / Food
    "chef":                            "CHEF",
    "cook":                            "CHEF",

    # Aviation
    "pilot":                           "AVIATION",
    "flight dispatcher":               "AVIATION",
    "aviation":                        "AVIATION",

    # Agriculture
    "agronomist":                      "AGRICULTURE",
    "farm manager":                    "AGRICULTURE",
    "agricultural engineer":           "AGRICULTURE",
}


# Category mapping for UpdatedResumeDataSet.csv (already used in merge_datasets.py)
UPDATED_MAP: dict[str, str] = {
    "Java Developer":            "INFORMATION-TECHNOLOGY",
    "Testing":                   "INFORMATION-TECHNOLOGY",
    "DevOps Engineer":           "INFORMATION-TECHNOLOGY",
    "Python Developer":          "INFORMATION-TECHNOLOGY",
    "Web Designing":             "DESIGNER",
    "HR":                        "HR",
    "Hadoop":                    "INFORMATION-TECHNOLOGY",
    "Data Science":              "INFORMATION-TECHNOLOGY",
    "Mechanical Engineer":       "ENGINEERING",
    "Sales":                     "SALES",
    "Operations Manager":        "BUSINESS-DEVELOPMENT",
    "ETL Developer":             "INFORMATION-TECHNOLOGY",
    "Blockchain":                "INFORMATION-TECHNOLOGY",
    "Arts":                      "ARTS",
    "Database":                  "INFORMATION-TECHNOLOGY",
    "Health and fitness":        "FITNESS",
    "Electrical Engineering":    "ENGINEERING",
    "PMO":                       "BUSINESS-DEVELOPMENT",
    "Business Analyst":          "BUSINESS-DEVELOPMENT",
    "DotNet Developer":          "INFORMATION-TECHNOLOGY",
    "Automation Testing":        "INFORMATION-TECHNOLOGY",
    "Network Security Engineer": "INFORMATION-TECHNOLOGY",
    "Civil Engineer":            "ENGINEERING",
    "SAP Developer":             "INFORMATION-TECHNOLOGY",
    "Advocate":                  "ADVOCATE",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_list_str(val) -> str:
    """Convert a Python-list-like string or NaN to a plain space-joined string."""
    if not isinstance(val, str) or not val.strip():
        return ""
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return " ".join(str(v) for v in parsed if v and str(v) != "None")
    except Exception:
        pass
    return val.strip()


def map_category(raw) -> "str | None":
    """Fuzzy-map a job title to one of the 24 model categories."""
    if not isinstance(raw, str):
        return None
    lower = raw.strip().lower()
    # Exact match first
    if lower in JOB_MAP:
        return JOB_MAP[lower]
    # Substring match (longer keys first to avoid false matches)
    for key in sorted(JOB_MAP, key=len, reverse=True):
        if key in lower:
            return JOB_MAP[key]
    return None


def build_feature(row: pd.Series) -> str:
    """Combine several columns into one rich text feature."""
    parts = [
        str(row.get("career_objective") or ""),
        safe_list_str(row.get("skills")),
        str(row.get("responsibilities") or ""),
        safe_list_str(row.get("positions")),
    ]
    combined = " ".join(p for p in parts if p.strip())
    return re.sub(r"\s+", " ", combined).strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parts: list[pd.DataFrame] = []

    # ── Source 1: resume_data.csv (new) ──────────────────────────────────────
    if SRC_NEW.exists():
        print(f"[*] Loading {SRC_NEW.name}  ({SRC_NEW.stat().st_size // 1024} KB) ...")
        df = pd.read_csv(SRC_NEW, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        print(f"    Shape: {df.shape}")

        df["Category"] = df["job_position_name"].apply(map_category)
        n_drop = df["Category"].isna().sum()
        df = df.dropna(subset=["Category"])
        print(f"    Unmapped (dropped): {n_drop}  |  Kept: {len(df)}")

        df["Feature"] = df.apply(build_feature, axis=1)
        df = df[df["Feature"].str.strip() != ""]

        src1 = df[["Feature", "Category"]].copy()
        print(f"    Category dist:\n{src1['Category'].value_counts().to_string()}")
        parts.append(src1)
    else:
        print(f"[!] {SRC_NEW.name} not found – skipping.")

    # ── Source 2: clean_resume_data.csv ──────────────────────────────────────
    if SRC_CLEAN.exists():
        base = pd.read_csv(SRC_CLEAN)
        cols = [c for c in ["Feature", "Category"] if c in base.columns]
        base = base[cols].dropna()
        print(f"\n[*] Loaded {SRC_CLEAN.name}: {len(base)} rows")
        parts.append(base)
    else:
        print(f"\n[!] {SRC_CLEAN.name} not found – skipping.")

    # ── Source 3: UpdatedResumeDataSet.csv ───────────────────────────────────
    if SRC_UPDATED.exists():
        extra = pd.read_csv(SRC_UPDATED)
        extra["Category"] = extra["Category"].map(UPDATED_MAP)
        extra = extra.rename(columns={"Resume": "Feature"})
        extra = extra[["Feature", "Category"]].dropna()
        print(f"[*] Loaded {SRC_UPDATED.name}: {len(extra)} rows")
        parts.append(extra)
    else:
        print(f"[!] {SRC_UPDATED.name} not found – skipping.")

    if not parts:
        print("[!] No source data found. Exiting.")
        sys.exit(1)

    # ── Combine, deduplicate, save ────────────────────────────────────────────
    print("\n[*] Combining all sources ...")
    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.dropna(subset=["Feature", "Category"])
    combined = combined[combined["Feature"].str.strip() != ""]
    combined = combined.drop_duplicates(subset=["Feature"])
    print(f"    Rows before dedup : {before}")
    print(f"    Rows after  dedup : {len(combined)}")

    combined.to_csv(DEST, index=False)
    print(f"\n[+] Saved: {DEST}  (total rows: {len(combined)})")
    print("\n[+] Final Category distribution:")
    print(combined["Category"].value_counts().to_string())
    print("\nDone! You can now run:  python train_model.py")


if __name__ == "__main__":
    main()
