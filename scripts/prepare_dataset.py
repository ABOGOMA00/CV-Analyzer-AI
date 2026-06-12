"""
prepare_dataset.py
==================
Builds clean_resume_data.csv from all available source CSVs.
Maps diverse category names to the project's taxonomy and
adds synthetic samples to fix the DESIGNER vs IT confusion.

Usage:
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "clean_resume_data.csv"

# ── Category normalisation ─────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str] = {
    # IT — all sub-types collapse to INFORMATION-TECHNOLOGY
    "information technology":   "INFORMATION-TECHNOLOGY",
    "java developer":           "INFORMATION-TECHNOLOGY",
    "python developer":         "INFORMATION-TECHNOLOGY",
    "dotnet developer":         "INFORMATION-TECHNOLOGY",
    "sap developer":            "INFORMATION-TECHNOLOGY",
    "etl developer":            "INFORMATION-TECHNOLOGY",
    "devops engineer":          "INFORMATION-TECHNOLOGY",
    "network security engineer":"INFORMATION-TECHNOLOGY",
    "database":                 "INFORMATION-TECHNOLOGY",
    "hadoop":                   "INFORMATION-TECHNOLOGY",
    "blockchain":               "INFORMATION-TECHNOLOGY",
    "web designing":            "DESIGNER",
    "testing":                  "INFORMATION-TECHNOLOGY",
    "automation testing":       "INFORMATION-TECHNOLOGY",
    "data science":             "INFORMATION-TECHNOLOGY",
    "it":                       "INFORMATION-TECHNOLOGY",
    "software":                 "INFORMATION-TECHNOLOGY",
    # Business
    "business development":     "BUSINESS-DEVELOPMENT",
    "business analyst":         "CONSULTANT",
    "pmo":                      "CONSULTANT",
    "operations manager":       "CONSULTANT",
    "sales":                    "SALES",
    "consultant":               "CONSULTANT",
    "consulting":               "CONSULTANT",
    "finance":                  "FINANCE",
    "banking":                  "BANKING",
    "accountant":               "ACCOUNTANT",
    "accounting":               "ACCOUNTANT",
    # People
    "hr":                       "HR",
    "human resources":          "HR",
    "teacher":                  "TEACHER",
    "teaching":                 "TEACHER",
    "education":                "TEACHER",
    # Health & Ops
    "healthcare":               "HEALTHCARE",
    "health and fitness":       "HEALTHCARE",
    "fitness":                  "FITNESS",
    "chef":                     "CHEF",
    "aviation":                 "AVIATION",
    "agriculture":              "AGRICULTURE",
    "automobile":               "AUTOMOBILE",
    "mechanical engineer":      "ENGINEERING",
    "civil engineer":           "ENGINEERING",
    "electrical engineering":   "ENGINEERING",
    "engineering":              "ENGINEERING",
    # Creative / Media
    "designer":                 "DESIGNER",
    "design":                   "DESIGNER",
    "digital media":            "DIGITAL-MEDIA",
    "arts":                     "ARTS",
    "public relations":         "PUBLIC-RELATIONS",
    # Other
    "advocate":                 "ADVOCATE",
    "construction":             "CONSTRUCTION",
    "bpo":                      "BPO",
    "apparel":                  "APPAREL",
}


def normalise_category(raw: str) -> str:
    key = str(raw).strip().lower()
    return CATEGORY_MAP.get(key, key.upper().replace(" ", "-"))


# ── Synthetic samples to fix DESIGNER vs INFORMATION-TECHNOLOGY confusion ──────
SYNTHETIC: list[tuple[str, str]] = [
    ("INFORMATION-TECHNOLOGY",
     "Frontend Developer 4 years experience React TypeScript JavaScript HTML5 CSS3 Redux "
     "Webpack Vite REST API Jest unit testing Git Agile Scrum e-commerce dashboard UIs "
     "responsive web applications Vue Angular Next.js performance optimisation CI/CD Docker AWS"),
    ("INFORMATION-TECHNOLOGY",
     "Backend Developer Python FastAPI Django Flask microservices Docker Kubernetes PostgreSQL "
     "MySQL Redis MongoDB CI/CD Jenkins GitHub Actions AWS Lambda EC2 REST API design "
     "authentication JWT message queue Kafka Elasticsearch TDD test-driven development"),
    ("INFORMATION-TECHNOLOGY",
     "Full Stack Developer Next.js React Node.js Express PostgreSQL MongoDB REST API JWT "
     "TypeScript Cypress testing AWS Vercel deployment Git version control Agile sprint "
     "frontend backend database scalable cloud-native applications"),
    ("INFORMATION-TECHNOLOGY",
     "Mobile Developer Flutter Dart React Native iOS Android Firebase App Store Google Play "
     "REST API Agile cross-platform fintech healthcare applications 50K users Kotlin Swift"),
    ("INFORMATION-TECHNOLOGY",
     "DevOps Engineer Docker Kubernetes Terraform Ansible AWS Azure GCP CI/CD GitHub Actions "
     "Jenkins Prometheus Grafana ELK stack Linux Bash scripting infrastructure automation "
     "monitoring cloud deployment scalability reliability"),
    ("INFORMATION-TECHNOLOGY",
     "Data Scientist Python Pandas NumPy Scikit-learn TensorFlow PyTorch machine learning "
     "deep learning NLP Hugging Face transformers classification regression clustering "
     "feature engineering Tableau Power BI Jupyter notebooks model deployment MLOps"),
    ("INFORMATION-TECHNOLOGY",
     "Software Engineer algorithms data structures system design object-oriented programming "
     "design patterns agile scrum code review unit testing debugging Python Java C++ SQL "
     "databases REST APIs Git microservices cloud AWS performance optimization"),
    ("DESIGNER",
     "UI UX Designer Figma Sketch Adobe XD Photoshop Illustrator user research usability "
     "testing A/B testing wireframes interactive prototypes design systems information "
     "architecture stakeholder collaboration product managers developers Agile"),
    ("DESIGNER",
     "Graphic Designer Adobe Creative Suite Photoshop Illustrator InDesign branding logo "
     "design typography print media visual identity color theory layout packaging "
     "motion graphics After Effects Premiere Pro 30 brands"),
    ("DESIGNER",
     "Product Designer design thinking user journey persona prototyping interaction design "
     "Figma design strategy cross-functional stakeholder collaboration high-fidelity "
     "mockups usability research UX writing accessibility standards"),
    ("FINANCE",
     "Financial Analyst financial modeling Excel DCF valuation budgeting forecasting "
     "variance analysis KPI dashboard reporting financial statements P&L balance sheet "
     "Bloomberg CFA investment banking private equity PowerPoint presentations"),
    ("HR",
     "HR Business Partner recruitment talent acquisition sourcing onboarding payroll HRIS "
     "performance management employee relations organizational development workforce "
     "planning succession planning change management Workday SAP SuccessFactors"),
]


def load_sources() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skip = {OUTPUT_CSV.name}

    for csv_file in RAW_DIR.glob("*.csv"):
        if csv_file.name in skip:
            continue
        print(f"[*] {csv_file.name} ...", end=" ")
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(csv_file, encoding="latin-1", on_bad_lines="skip")
            except Exception as e:
                print(f"SKIP ({e})")
                continue

        cat_col  = next((c for c in df.columns if c.lower() in ("category", "label", "class")), None)
        text_col = next((c for c in df.columns if c.lower() in ("feature", "resume", "text", "cv", "content")), None)

        if not cat_col or not text_col:
            print("SKIP (no category/text cols)")
            continue

        sub = df[[cat_col, text_col]].rename(columns={cat_col: "Category", text_col: "Feature"})
        sub = sub.dropna()
        sub["Category"] = sub["Category"].apply(normalise_category)
        frames.append(sub)
        print(f"{len(sub):,} rows, {sub['Category'].nunique()} cats")

    if not frames:
        return pd.DataFrame(columns=["Category", "Feature"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("\n" + "=" * 60)
    print("  CV ANALYZER AI - DATASET PREPARATION")
    print("=" * 60 + "\n")

    # 1. Load all source CSVs
    df = load_sources()
    print(f"\n[+] Source rows: {len(df):,}")

    # 2. Add synthetic samples
    synth = pd.DataFrame(SYNTHETIC, columns=["Category", "Feature"])
    df = pd.concat([df, synth], ignore_index=True)
    print(f"[+] After synthetic: {len(df):,}")

    # 3. Light cleanup — drop exact duplicates and very short texts only
    df["Feature"] = df["Feature"].astype(str).str.strip()
    df = df[df["Feature"].str.len() > 80]
    df = df.drop_duplicates(subset=["Feature"])
    print(f"[+] After cleanup:  {len(df):,}")

    # 4. Show distribution
    print("\n[+] Category distribution:")
    for cat, cnt in df["Category"].value_counts().items():
        bar = "#" * min(cnt, 50)
        print(f"    {cat:<30} {cnt:>4}  {bar}")

    # 5. Save
    df.insert(0, "ID", range(1, len(df) + 1))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] Saved {len(df):,} rows -> {OUTPUT_CSV}")
    print("[>>] Next: run train_model.py")


if __name__ == "__main__":
    main()
