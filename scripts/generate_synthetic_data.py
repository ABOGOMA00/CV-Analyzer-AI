"""
generate_synthetic_data.py
==========================
Generates diverse, high-quality synthetic resume samples for each
project category to supplement the limited real data.

Produces ~80-120 samples per category using varied templates and
keyword combinations, then merges with real data and saves the
final clean_resume_data.csv ready for train_model.py.

Usage:
    python scripts/generate_synthetic_data.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
import pandas as pd

random.seed(42)
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "clean_resume_data.csv"

# ── Template library ───────────────────────────────────────────────────────────
# Each category has: intro templates + skill pools + achievement pools
# generate() mixes them to create diverse, non-duplicate samples.

TEMPLATES: dict[str, dict] = {

    "INFORMATION-TECHNOLOGY": {
        "titles": [
            "Software Engineer", "Frontend Developer", "Backend Developer",
            "Full Stack Developer", "Data Engineer", "DevOps Engineer",
            "Mobile Developer", "Cloud Architect", "ML Engineer",
            "Software Developer", "Python Engineer", "Java Developer",
        ],
        "skills": [
            ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "REST API", "Git", "CI/CD"],
            ["React", "TypeScript", "JavaScript", "HTML5", "CSS3", "Webpack", "Redux", "Jest"],
            ["Java", "Spring Boot", "Microservices", "Kafka", "MySQL", "Kubernetes", "Jenkins"],
            ["Flutter", "Dart", "iOS", "Android", "Firebase", "React Native", "App Store"],
            ["Node.js", "Express", "MongoDB", "Redis", "GraphQL", "Next.js", "Vercel"],
            ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "MLOps"],
            ["Docker", "Kubernetes", "Terraform", "AWS", "Azure", "GCP", "Prometheus", "Grafana"],
            ["Vue.js", "Angular", "Nuxt.js", "SASS", "Bootstrap", "Tailwind", "Figma"],
            ["C#", ".NET", "ASP.NET", "Azure", "SQL Server", "Entity Framework", "WPF"],
            ["Go", "gRPC", "PostgreSQL", "Redis", "Docker", "Linux", "Bash", "API design"],
            ["PHP", "Laravel", "MySQL", "Vue.js", "RESTful API", "Git", "Nginx", "Redis"],
            ["Kotlin", "Android", "Jetpack Compose", "MVVM", "Retrofit", "Room", "Coroutines"],
            ["Swift", "SwiftUI", "iOS", "Xcode", "Core Data", "ARKit", "TestFlight", "Firebase"],
        ],
        "achievements": [
            "Reduced API response time by {n}% through query optimisation.",
            "Led development of {n} microservices serving {m}K+ daily requests.",
            "Improved test coverage from {n}% to {m}% using TDD practices.",
            "Deployed containerised applications handling {n}M+ monthly users.",
            "Architected CI/CD pipeline reducing deployment time from {n} hours to {m} minutes.",
            "Built real-time dashboard processing {n}K events per second.",
            "Mentored team of {n} junior developers across {m} agile sprints.",
            "Migrated monolith to microservices, cutting infrastructure cost by {n}%.",
            "Implemented OAuth2 and JWT auth system securing {n}K+ user accounts.",
            "Optimised database queries improving throughput by {n}%.",
        ],
    },

    "FINANCE": {
        "titles": [
            "Financial Analyst", "Senior Financial Analyst", "Finance Manager",
            "Investment Analyst", "Portfolio Analyst", "Risk Analyst",
            "Budget Analyst", "FP&A Analyst", "Treasury Analyst",
        ],
        "skills": [
            ["Financial Modeling", "Excel", "DCF", "Valuation", "Bloomberg", "Forecasting"],
            ["Budgeting", "FP&A", "Variance Analysis", "KPIs", "PowerPoint", "SAP"],
            ["Risk Management", "VAR", "Stress Testing", "Basel III", "Compliance", "Audit"],
            ["Investment Analysis", "Equity Research", "CFA", "Capital Markets", "M&A"],
            ["Financial Reporting", "IFRS", "GAAP", "Quickbooks", "Oracle", "Consolidation"],
            ["Cash Flow Modeling", "LBO", "Pitch Deck", "Private Equity", "Due Diligence"],
        ],
        "achievements": [
            "Built {n}-year financial model forecasting $${m}M revenue with {p}% accuracy.",
            "Reduced budget variance to under {n}% across all departments.",
            "Managed investment portfolio worth $${n}M achieving {m}% annual return.",
            "Automated monthly reporting, saving {n} analyst-hours per quarter.",
            "Identified cost savings of $${n}M through detailed variance analysis.",
            "Prepared financial presentations for board and C-suite stakeholders.",
            "Led due diligence for {n} M&A transactions totalling $${m}B.",
        ],
    },

    "HR": {
        "titles": [
            "HR Business Partner", "Talent Acquisition Specialist", "HR Manager",
            "Recruitment Manager", "People Operations Lead", "HR Generalist",
            "Compensation & Benefits Analyst", "Learning & Development Manager",
        ],
        "skills": [
            ["Recruitment", "Talent Acquisition", "Sourcing", "LinkedIn Recruiter", "ATS"],
            ["Performance Management", "Employee Relations", "HRBP", "Succession Planning"],
            ["Payroll", "HRIS", "Workday", "SAP SuccessFactors", "ADP", "Compensation"],
            ["Onboarding", "Employer Branding", "Candidate Experience", "Job Posting"],
            ["Training & Development", "L&D", "LMS", "Moodle", "e-Learning", "Coaching"],
            ["HR Analytics", "People Data", "Dashboards", "Attrition Analysis", "Workforce Planning"],
        ],
        "achievements": [
            "Reduced time-to-hire from {n} to {m} days across all departments.",
            "Sourced and placed {n}+ candidates in {m} months with {p}% retention rate.",
            "Designed onboarding programme reducing new-hire ramp time by {n}%.",
            "Managed payroll for {n}+ employees across {m} countries.",
            "Built talent pipeline resulting in {n}% reduction in agency spend.",
            "Implemented performance review cycle for {n}+ employees using Workday.",
            "Achieved employee satisfaction score of {n}% in annual engagement survey.",
        ],
    },

    "DESIGNER": {
        "titles": [
            "UI/UX Designer", "Product Designer", "Graphic Designer",
            "Visual Designer", "Interaction Designer", "Brand Designer",
            "Motion Designer", "UX Researcher",
        ],
        "skills": [
            ["Figma", "Sketch", "Adobe XD", "Prototyping", "Wireframing", "Design Systems"],
            ["Photoshop", "Illustrator", "InDesign", "Branding", "Typography", "Color Theory"],
            ["User Research", "Usability Testing", "A/B Testing", "User Journeys", "Personas"],
            ["After Effects", "Premiere Pro", "Motion Graphics", "Animation", "Storyboarding"],
            ["Design Thinking", "Information Architecture", "Accessibility", "WCAG", "UX Writing"],
        ],
        "achievements": [
            "Redesigned onboarding flow increasing user activation by {n}%.",
            "Created design system with {n}+ reusable components adopted by {m} teams.",
            "Conducted {n} user interviews leading to {m}% improvement in task completion.",
            "Reduced customer support tickets by {n}% through UX improvements.",
            "Delivered brand identity for {n} clients across {m} industries.",
            "Improved app store rating from {p}.{q} to {r}.{s} through UX redesign.",
        ],
    },

    "HEALTHCARE": {
        "titles": [
            "Registered Nurse", "Clinical Nurse Specialist", "Healthcare Manager",
            "Medical Officer", "Pharmacist", "Physical Therapist", "Lab Technician",
        ],
        "skills": [
            ["Patient Care", "Clinical Skills", "EMR", "EHR", "Epic", "CPR", "ACLS"],
            ["Nursing", "Medication Administration", "Vital Signs", "Wound Care", "Triage"],
            ["Medical Documentation", "HIPAA Compliance", "ICD-10", "Clinical Protocols"],
            ["Pharmacy", "Drug Interactions", "Dispensing", "Clinical Pharmacy", "Formulary"],
            ["Physical Therapy", "Rehabilitation", "Exercise Prescription", "Patient Assessment"],
        ],
        "achievements": [
            "Managed care for {n}+ patients per shift in high-acuity unit.",
            "Reduced medication errors by {n}% through double-check protocol.",
            "Achieved patient satisfaction score of {n}% for {m} consecutive months.",
            "Trained {n} junior nurses on EMR system and clinical protocols.",
            "Implemented fall prevention programme reducing incidents by {n}%.",
        ],
    },

    "SALES": {
        "titles": [
            "Account Executive", "Sales Manager", "Business Development Manager",
            "Sales Representative", "Regional Sales Manager", "Enterprise Account Manager",
        ],
        "skills": [
            ["CRM", "Salesforce", "HubSpot", "Cold Calling", "Lead Generation", "Pipeline"],
            ["B2B Sales", "Negotiation", "Closing Deals", "Quota Achievement", "Upselling"],
            ["Account Management", "Revenue Growth", "Territory Management", "Forecasting"],
            ["Prospecting", "LinkedIn Sales Navigator", "Demo Presentation", "Proposal Writing"],
        ],
        "achievements": [
            "Exceeded annual quota by {n}% generating $${m}M in new revenue.",
            "Closed {n} enterprise deals averaging $${m}K ARR within {p} months.",
            "Built pipeline of {n} qualified opportunities worth $${m}M.",
            "Ranked #{n} out of {m} sales reps nationally for {p} consecutive quarters.",
            "Grew territory revenue from $${n}M to $${m}M in {p} months.",
        ],
    },

    "BANKING": {
        "titles": [
            "Credit Analyst", "Relationship Manager", "Compliance Officer",
            "Risk Manager", "Investment Associate", "Branch Manager",
        ],
        "skills": [
            ["Credit Analysis", "Loan Underwriting", "Financial Statements", "Risk Assessment"],
            ["AML", "KYC", "Compliance", "Regulatory Reporting", "FINRA", "Basel III"],
            ["Investment Banking", "M&A", "Capital Markets", "Bloomberg", "CFA", "Pitch Deck"],
            ["Retail Banking", "Customer Service", "Cross-selling", "Account Management"],
        ],
        "achievements": [
            "Managed credit portfolio of $${n}M with NPL ratio below {m}%.",
            "Processed {n}+ loan applications monthly with {m}% approval accuracy.",
            "Ensured {n}% compliance rate across {m} regulatory audits.",
            "Generated $${n}M in fee income through cross-selling products.",
        ],
    },

    "ACCOUNTANT": {
        "titles": [
            "Financial Accountant", "Tax Accountant", "Senior Accountant",
            "Audit Manager", "Management Accountant", "CPA",
        ],
        "skills": [
            ["GAAP", "IFRS", "Financial Reporting", "General Ledger", "Reconciliation"],
            ["Tax Preparation", "CPA", "IRS", "VAT", "Corporate Tax", "Tax Returns"],
            ["Internal Audit", "External Audit", "SOX", "Risk Assessment", "Audit Planning"],
            ["QuickBooks", "SAP", "Oracle", "Accounts Payable", "Accounts Receivable"],
        ],
        "achievements": [
            "Managed month-end close process for {n}+ entity company reducing cycle by {m} days.",
            "Prepared financial statements with {n}M in assets under management.",
            "Identified $${n}K in tax savings through strategic planning.",
            "Led audit team of {n} ensuring {m}% compliance across all departments.",
        ],
    },

    "ENGINEERING": {
        "titles": [
            "Mechanical Engineer", "Civil Engineer", "Electrical Engineer",
            "Structural Engineer", "Project Engineer", "Chemical Engineer",
        ],
        "skills": [
            ["AutoCAD", "SolidWorks", "ANSYS", "Mechanical Design", "FEA", "GD&T", "CATIA"],
            ["Structural Design", "Revit", "STAAD", "Concrete", "Steel", "BIM", "Construction"],
            ["PLC", "SCADA", "Circuit Design", "Embedded Systems", "MATLAB", "Power Systems"],
            ["Project Management", "Quality Control", "ISO 9001", "Safety Engineering", "HAZOP"],
        ],
        "achievements": [
            "Designed structural components for {n} commercial projects totalling $${m}M.",
            "Reduced manufacturing defects by {n}% through process improvement.",
            "Led {n}-person engineering team delivering project {m}% under budget.",
            "Implemented quality management system achieving ISO {n} certification.",
        ],
    },

    "TEACHER": {
        "titles": [
            "High School Teacher", "University Lecturer", "Online Educator",
            "Primary School Teacher", "Curriculum Developer", "Academic Coordinator",
        ],
        "skills": [
            ["Curriculum Design", "Lesson Planning", "Classroom Management", "Assessment"],
            ["IB", "IGCSE", "Differentiated Instruction", "Special Needs", "IEP"],
            ["e-Learning", "LMS", "Moodle", "Canvas", "Online Teaching", "Zoom"],
            ["Research", "Publication", "Thesis Supervision", "Academic Writing", "Grant"],
        ],
        "achievements": [
            "Improved student pass rates from {n}% to {m}% over {p} academic year.",
            "Developed curriculum adopted by {n} schools across {m} districts.",
            "Taught {n}+ students online achieving {m}% satisfaction rating.",
            "Published {n} peer-reviewed papers in {m} international journals.",
        ],
    },

    "CONSULTANT": {
        "titles": [
            "Management Consultant", "IT Consultant", "Strategy Consultant",
            "Business Analyst", "Financial Consultant", "Operations Consultant",
        ],
        "skills": [
            ["Strategy", "Business Analysis", "Process Improvement", "Change Management"],
            ["Stakeholder Management", "Project Management", "Presentations", "Case Studies"],
            ["ERP", "SAP", "Salesforce", "Digital Transformation", "Requirements Gathering"],
            ["Financial Advisory", "Wealth Management", "CFP", "Portfolio Management"],
        ],
        "achievements": [
            "Delivered $${n}M cost savings through process re-engineering for {m} clients.",
            "Led digital transformation for {n} clients across {m} industries.",
            "Built business case for $${n}M investment approved by C-suite.",
            "Managed {n}+ stakeholders across {m} business units simultaneously.",
        ],
    },

    "DIGITAL-MEDIA": {
        "titles": [
            "SEO Specialist", "Digital Marketing Manager", "Social Media Manager",
            "Content Creator", "PPC Specialist", "Growth Marketer",
        ],
        "skills": [
            ["SEO", "Google Analytics", "Ahrefs", "SEMrush", "Keyword Research", "Link Building"],
            ["Google Ads", "Facebook Ads", "PPC", "Meta Ads", "LinkedIn Ads", "TikTok Ads"],
            ["Social Media", "Instagram", "Content Calendar", "Community Management", "Engagement"],
            ["Email Marketing", "Mailchimp", "HubSpot", "Marketing Automation", "CRM", "Funnel"],
            ["Content Creation", "Copywriting", "Blog", "YouTube", "Podcast", "Brand Voice"],
        ],
        "achievements": [
            "Grew organic traffic by {n}% through SEO strategy within {m} months.",
            "Managed $${n}K monthly ad spend achieving {m}% ROAS.",
            "Grew social following from {n}K to {m}K in {p} months.",
            "Increased email open rate from {n}% to {m}% through A/B testing.",
            "Generated {n}+ leads per month at $${m} cost per lead.",
        ],
    },

    "AVIATION": {
        "titles": ["Commercial Pilot", "First Officer", "Flight Dispatcher", "Aviation Safety Officer"],
        "skills": [
            ["ATPL", "CPL", "IFR", "VFR", "Multi-engine", "B737", "A320", "Flight Hours"],
            ["Flight Dispatch", "Flight Planning", "Weather Analysis", "NOTAM", "Fuel Calculation"],
            ["Safety Management", "SMS", "ICAO", "FAA", "EASA", "CRM", "Threat and Error Management"],
        ],
        "achievements": [
            "Logged {n}+ flight hours across {m} aircraft types.",
            "Maintained {n}% on-time departure rate over {m} consecutive months.",
            "Conducted {n} safety audits with zero critical findings.",
        ],
    },

    "AGRICULTURE": {
        "titles": ["Agronomist", "Farm Manager", "Agricultural Engineer", "Crop Scientist"],
        "skills": [
            ["Agronomy", "Soil Science", "Crop Management", "Irrigation", "Fertilization", "GIS"],
            ["Farm Management", "Livestock", "Harvest", "Pest Control", "Organic Farming"],
            ["Precision Agriculture", "Drone", "IoT Sensors", "Data Analysis", "Automation"],
        ],
        "achievements": [
            "Increased crop yield by {n}% through optimised irrigation scheduling.",
            "Reduced pesticide usage by {n}% implementing integrated pest management.",
            "Managed {n}-hectare farm operation with annual revenue of $${m}M.",
        ],
    },

    "CHEF": {
        "titles": ["Executive Chef", "Sous Chef", "Pastry Chef", "Line Cook", "Head Chef"],
        "skills": [
            ["Kitchen Management", "Menu Development", "Food Cost Control", "HACCP", "Food Safety"],
            ["Fine Dining", "Culinary Arts", "Team Leadership", "Inventory Management", "Purchasing"],
            ["Pastry", "Baking", "Dessert", "Chocolate", "Bread Making", "Confectionery"],
        ],
        "achievements": [
            "Managed kitchen team of {n} producing {m}+ covers per service.",
            "Reduced food cost from {n}% to {m}% through portion control and menu engineering.",
            "Developed seasonal menu increasing average spend per head by {n}%.",
        ],
    },

    "BUSINESS-DEVELOPMENT": {
        "titles": ["Business Development Manager", "BD Executive", "Partnership Manager", "Growth Manager"],
        "skills": [
            ["Lead Generation", "Partnerships", "Market Expansion", "Negotiation", "Strategic Alliances"],
            ["CRM", "HubSpot", "Salesforce", "Pipeline Management", "Revenue Growth", "B2B"],
            ["Market Research", "Competitive Analysis", "Go-to-Market", "Contract Management"],
        ],
        "achievements": [
            "Secured {n} strategic partnerships generating $${m}M in incremental revenue.",
            "Expanded into {n} new markets within {m} months of joining.",
            "Built BD pipeline of $${n}M closing {m}% within the first year.",
        ],
    },

    "ARTS": {
        "titles": ["Artist", "Musician", "Performer", "Theatre Director", "Visual Artist"],
        "skills": [
            ["Fine Arts", "Visual Arts", "Painting", "Sculpture", "Exhibition", "Curation"],
            ["Music Performance", "Composition", "Music Theory", "Orchestra", "Recording Studio"],
            ["Theatre", "Acting", "Directing", "Stage Management", "Script Writing", "Production"],
        ],
        "achievements": [
            "Exhibited artwork in {n} national and international galleries.",
            "Performed in {n} productions across {m} countries.",
            "Composed original score for {n} films and theatre productions.",
        ],
    },

    "PUBLIC-RELATIONS": {
        "titles": ["PR Manager", "Communications Manager", "Media Relations Specialist", "PR Executive"],
        "skills": [
            ["Media Relations", "Press Release", "Crisis Communication", "Brand Reputation", "PR Strategy"],
            ["Social Media", "Content Strategy", "Influencer Relations", "Event Management", "Journalism"],
        ],
        "achievements": [
            "Secured {n}+ media placements in top-tier publications.",
            "Managed crisis communication reducing negative sentiment by {n}%.",
            "Organised {n} press events reaching {m}M+ audience reach.",
        ],
    },

    "ADVOCATE": {
        "titles": ["Lawyer", "Attorney", "Legal Counsel", "Corporate Lawyer", "Litigation Specialist"],
        "skills": [
            ["Legal Research", "Contract Law", "Litigation", "Corporate Law", "Compliance", "Due Diligence"],
            ["Negotiation", "Mediation", "Arbitration", "Intellectual Property", "Employment Law"],
        ],
        "achievements": [
            "Successfully litigated {n} cases with {m}% win rate.",
            "Drafted and reviewed {n}+ contracts totalling $${m}M in value.",
            "Provided legal counsel for {n} corporate transactions.",
        ],
    },

    "CONSTRUCTION": {
        "titles": ["Construction Manager", "Site Manager", "Project Manager", "Civil Contractor"],
        "skills": [
            ["Construction Management", "Site Supervision", "AutoCAD", "MS Project", "Budget Control"],
            ["Health and Safety", "ISO 45001", "Risk Assessment", "BIM", "Procurement", "Subcontractor Management"],
        ],
        "achievements": [
            "Delivered {n} construction projects on time and {m}% under budget.",
            "Managed site workforce of {n}+ workers maintaining zero lost-time incidents.",
            "Oversaw $${n}M construction project from groundbreaking to completion.",
        ],
    },

    "BPO": {
        "titles": ["BPO Team Leader", "Customer Service Representative", "Process Analyst", "Quality Analyst"],
        "skills": [
            ["Customer Service", "Call Center", "CRM", "CSAT", "NPS", "SLA", "Escalation Management"],
            ["Process Improvement", "Six Sigma", "Quality Assurance", "Workforce Management", "Zendesk"],
        ],
        "achievements": [
            "Maintained CSAT score of {n}% for {m} consecutive months.",
            "Reduced average handle time from {n} to {m} minutes.",
            "Managed team of {n} agents achieving {m}% SLA compliance.",
        ],
    },
}


def _fill(template: str) -> str:
    """Replace {n}, {m}, {p}, etc. with random realistic numbers."""
    values = {
        "n": random.choice([15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 85, 90, 95]),
        "m": random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 50, 100]),
        "p": random.choice([3, 6, 9, 12, 18, 24, 30]),
        "q": random.choice([1, 2, 3, 4, 5]),
        "r": random.choice([4, 5]),
        "s": random.choice([2, 3, 4, 5, 6, 7, 8]),
    }
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def generate_sample(category: str, data: dict, idx: int) -> str:
    """Build one diverse resume sample."""
    title = random.choice(data["titles"])
    skills = random.choice(data["skills"])
    yoe = random.randint(1, 12)
    extra_skills = random.sample(
        [s for pool in data["skills"] for s in pool if s not in skills], k=min(4, sum(len(p) for p in data["skills"]) - len(skills))
    )
    combined_skills = skills + extra_skills[:3]
    random.shuffle(combined_skills)

    n_achievements = random.randint(2, 4)
    achievements = random.sample(data["achievements"], min(n_achievements, len(data["achievements"])))
    achievement_text = " ".join(_fill(a) for a in achievements)

    edu_choices = [
        f"Bachelor's in {random.choice(['Business', 'Computer Science', 'Engineering', 'Finance', 'Economics', 'Arts'])}.",
        f"Master's in {random.choice(['Business Administration', 'Data Science', 'Finance', 'Engineering', 'Education'])}.",
        f"BSc {random.choice(['Computer Science', 'Finance', 'Accounting', 'Nursing', 'Civil Engineering'])}.",
    ]
    edu = random.choice(edu_choices)

    skill_str = ", ".join(combined_skills)

    templates = [
        f"{title} with {yoe}+ years of experience. Skills: {skill_str}. {achievement_text} Education: {edu}",
        f"Experienced {title} specialising in {', '.join(skills[:3])}. {yoe} years in the industry. {achievement_text} Proficient in {skill_str}. {edu}",
        f"Professional {title} — {yoe} years experience. Technical expertise: {skill_str}. Key achievements: {achievement_text} {edu}",
        f"{title} | {yoe} Years Experience | {skills[0]} | {skills[1] if len(skills)>1 else ''}. Core competencies: {skill_str}. {achievement_text} {edu}",
    ]
    return random.choice(templates)


def main() -> None:
    print("\n" + "=" * 60)
    print("  CV ANALYZER AI - SYNTHETIC DATA GENERATOR")
    print("=" * 60 + "\n")

    TARGET_PER_CATEGORY = 100  # aim for ~100 samples per category
    rows: list[dict] = []

    for category, data in TEMPLATES.items():
        samples_needed = TARGET_PER_CATEGORY
        generated: set[str] = set()
        attempts = 0

        while len(generated) < samples_needed and attempts < samples_needed * 5:
            sample = generate_sample(category, data, attempts)
            if sample not in generated:
                generated.add(sample)
                rows.append({"Category": category, "Feature": sample})
            attempts += 1

        print(f"  {category:<30} {len(generated):>4} samples")

    synth_df = pd.DataFrame(rows)
    print(f"\n[+] Total synthetic samples: {len(synth_df):,}")

    # Load real data (UpdatedResumeDataSet unique rows)
    real_frames = []
    skip = {OUTPUT_CSV.name}
    for csv_file in RAW_DIR.glob("*.csv"):
        if csv_file.name in skip:
            continue
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
            cat_col  = next((c for c in df.columns if c.lower() in ("category", "label", "class")), None)
            text_col = next((c for c in df.columns if c.lower() in ("feature", "resume", "text", "cv", "content")), None)
            if cat_col and text_col:
                sub = df[[cat_col, text_col]].rename(columns={cat_col: "Category", text_col: "Feature"})
                sub = sub.dropna().drop_duplicates(subset=["Feature"])
                sub["Category"] = sub["Category"].apply(
                    lambda x: {"information technology": "INFORMATION-TECHNOLOGY",
                               "java developer": "INFORMATION-TECHNOLOGY",
                               "python developer": "INFORMATION-TECHNOLOGY",
                               "devops engineer": "INFORMATION-TECHNOLOGY",
                               "testing": "INFORMATION-TECHNOLOGY",
                               "automation testing": "INFORMATION-TECHNOLOGY",
                               "dotnet developer": "INFORMATION-TECHNOLOGY",
                               "sap developer": "INFORMATION-TECHNOLOGY",
                               "etl developer": "INFORMATION-TECHNOLOGY",
                               "hadoop": "INFORMATION-TECHNOLOGY",
                               "database": "INFORMATION-TECHNOLOGY",
                               "blockchain": "INFORMATION-TECHNOLOGY",
                               "network security engineer": "INFORMATION-TECHNOLOGY",
                               "web designing": "DESIGNER",
                               "health and fitness": "HEALTHCARE",
                               "mechanical engineer": "ENGINEERING",
                               "civil engineer": "ENGINEERING",
                               "electrical engineering": "ENGINEERING",
                               "business analyst": "CONSULTANT",
                               "operations manager": "CONSULTANT",
                               "pmo": "CONSULTANT",
                               "arts": "ARTS",
                               "advocate": "ADVOCATE",
                               "hr": "HR",
                               "sales": "SALES",
                               "finance": "FINANCE",
                               }.get(str(x).strip().lower(), str(x).strip().upper().replace(" ", "-"))
                )
                real_frames.append(sub)
                print(f"[+] Real data from {csv_file.name}: {len(sub):,} unique rows")
        except Exception as e:
            print(f"[!] Skipped {csv_file.name}: {e}")

    if real_frames:
        real_df = pd.concat(real_frames, ignore_index=True)
        real_df = real_df[real_df["Feature"].astype(str).str.len() > 80]
        combined = pd.concat([real_df, synth_df], ignore_index=True)
    else:
        combined = synth_df

    # Light dedup on exact text only
    combined = combined.drop_duplicates(subset=["Feature"])
    print(f"\n[+] Final combined: {len(combined):,} samples")

    print("\n[+] Category distribution:")
    for cat, cnt in combined["Category"].value_counts().items():
        bar = "#" * (cnt // 3)
        print(f"    {cat:<30} {cnt:>4}  {bar}")

    combined.insert(0, "ID", range(1, len(combined) + 1))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] Saved {len(combined):,} rows -> {OUTPUT_CSV}")
    print("[>>] Next: python train_model.py")


if __name__ == "__main__":
    main()
