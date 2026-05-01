"""
merge_datasets.py
=================
Legacy dataset merger. Superseded by scripts/prepare_resume_data.py for full merging.
Now updated to read from and write to the data/ folder structure.

Run from the project root:
    python scripts/merge_datasets.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

print("[*] Loading original dataset...")
df1 = pd.read_csv(RAW / "clean_resume_data.csv")
print(f"    Original shape: {df1.shape}")

print("[*] Loading new dataset...")
df2 = pd.read_csv(RAW / "UpdatedResumeDataSet.csv")
print(f"    New shape: {df2.shape}")

# Mapping new categories to existing 24 categories
category_mapping = {
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

print("[*] Mapping categories...")
df2["Category"] = df2["Category"].map(category_mapping)
df2 = df2.rename(columns={"Resume": "Feature"})
df2 = df2[["Category", "Feature"]]

if "ID" in df1.columns:
    df1 = df1[["Category", "Feature"]]

print("[*] Concatenating datasets...")
df_combined = pd.concat([df1, df2], ignore_index=True).dropna()

print(f"[+] Combined shape: {df_combined.shape}")
print("[*] New Category Distribution:")
print(df_combined["Category"].value_counts())

output_file = PROC / "merged_resume_data.csv"
df_combined.to_csv(output_file, index=False)
print(f"[+] Saved merged dataset to {output_file}")
