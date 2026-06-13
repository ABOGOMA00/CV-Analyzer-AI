# services/ml_service.py
# ============================================================
# Loads the ML pipeline and runs predictions.
# Also: ATS scoring, smart skill extraction, sub-specialization
# detection (Web Dev, Mobile, Desktop, Backend, Frontend, etc.)
# ============================================================

import os
import re
import joblib
import numpy as np
from pathlib import Path



# ── Optional heavy dependencies (graceful fallback) ───────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH") or next(
        (
            str(path)
            for path in (Path.home() / ".cache" / "huggingface" / "hub").glob(
                "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
            )
            if path.is_dir()
        ),
        "all-MiniLM-L6-v2",
    )
    _similarity_model = SentenceTransformer(_EMBED_MODEL_PATH)
    print("[+] SentenceTransformer loaded successfully")
except Exception as e:
    print(f"[!] SentenceTransformer not available: {e}")
    _similarity_model = None

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    print("[+] spaCy model loaded successfully")
except Exception as e:
    print(f"[!] spaCy not available: {e}")
    _nlp = None

# ── Classification model ──────────────────────────────────────────────────────
SAVED_MODEL_DIR = Path(__file__).parent.parent.parent / "saved_model"
MODEL_PATH      = SAVED_MODEL_DIR / "model.pkl"
ENCODER_PATH    = SAVED_MODEL_DIR / "label_encoder.pkl"
TFIDF_W_PATH    = SAVED_MODEL_DIR / "tfidf_word.pkl"
TFIDF_C_PATH    = SAVED_MODEL_DIR / "tfidf_char.pkl"
SCALER_PATH     = SAVED_MODEL_DIR / "scaler.pkl"

# Global model artifacts
model      = None
encoder    = None
tfidf_word = None
tfidf_char = None
scaler     = None

def _load_models():
    global model, encoder, tfidf_word, tfidf_char, scaler
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        if ENCODER_PATH.exists():
            encoder = joblib.load(ENCODER_PATH)
        if TFIDF_W_PATH.exists():
            tfidf_word = joblib.load(TFIDF_W_PATH)
        if TFIDF_C_PATH.exists():
            tfidf_char = joblib.load(TFIDF_C_PATH)
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            
        status = []
        if model: status.append("Model")
        if encoder: status.append(f"Encoder({len(encoder.classes_)})")
        if tfidf_word: status.append("TFIDF-W")
        if tfidf_char: status.append("TFIDF-C")
        if scaler: status.append("Scaler")
        
        if status:
            print(f"[+] Classifier artifacts loaded: {', '.join(status)}")
    except Exception as e:
        print(f"[x] Failed to load classifier artifacts: {e}")

_load_models()


def _is_text_pipeline(model_obj) -> bool:
    return hasattr(model_obj, "predict") and not hasattr(model_obj, "decision_function")


def _ensure_classifier_ready() -> None:
    if model is None or encoder is None:
        _load_models()
    if model is None or encoder is None:
        raise RuntimeError("Classifier artifacts are missing. Run train_model.py first.")


def _ensure_similarity_model():
    global _similarity_model
    if _similarity_model is not None:
        return _similarity_model
    try:
        from sentence_transformers import SentenceTransformer

        embed_model_path = os.getenv("EMBED_MODEL_PATH") or next(
            (
                str(path)
                for path in (Path.home() / ".cache" / "huggingface" / "hub").glob(
                    "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
                )
                if path.is_dir()
            ),
            "all-MiniLM-L6-v2",
        )
        _similarity_model = SentenceTransformer(embed_model_path)
        return _similarity_model
    except Exception as exc:
        raise RuntimeError(f"SentenceTransformer could not be loaded: {exc}") from exc

def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.atleast_2d(scores).astype(float)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_scores / denom


def _predict_proba_any(model_obj, x):
    if hasattr(model_obj, "predict_proba"):
        return model_obj.predict_proba(x)
    if hasattr(model_obj, "decision_function"):
        return _softmax_rows(model_obj.decision_function(x))
    raise RuntimeError("Model does not support predict_proba or decision_function.")


def _display_confidence(probas: np.ndarray) -> float:
    """
    LinearSVC margins are not calibrated probabilities. Convert the winner gap into
    a user-facing confidence that is easier to read in the UI.
    """
    values = np.asarray(probas, dtype=float)
    if values.size == 0:
        return 0.0
    ranked = np.sort(values)[::-1]
    top = float(ranked[0])
    second = float(ranked[1]) if ranked.size > 1 else 0.0
    gap = max(0.0, top - second)
    calibrated = 35.0 + (top * 180.0) + (gap * 420.0)
    return max(30.0, min(92.0, calibrated))


# ── Noise words to filter out of extracted skills ────────────────────────────
_NOISE_WORDS = {
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
    "january","february","march","april","june","july","august","september",
    "october","november","december","monday","tuesday","wednesday","thursday",
    "friday","saturday","sunday","bachelor","master","degree","university",
    "college","school","institute","faculty","gpa","cum","laude","major",
    "minor","department","mood","trio","shift","hands","approval","formula",
    "area","team","volunteer","systems","planning","the","and","for","with",
    "that","this","from","have","has","been","were","they","them","than",
    "also","into","about","which","when","more","some","such","each","both",
    "its","our","your","their","will","would","could","should","may","might",
    "very","just","even","well","much","also","here","there","where","while",
    "first","second","third","last","new","old","good","high","low","large",
    "small","big","many","few","all","any","one","two","three","four","five",
    "six","seven","eight","nine","ten","year","years","month","months","day",
    "days","week","weeks","time","work","working","worked","use","used","using",
    "make","made","making","build","built","building","get","got","getting",
    "help","helping","lead","leading","run","running","set","setting","take",
    "taking","give","given","come","coming","go","going","see","seen","know",
    "known","think","thought","look","looking","called","name","position",
    "role","job","company","organization","experience","skill","ability",
    "knowledge","background","profile","objective","summary","reference",
    "contact","address","phone","email","linkedin","github","portfolio",
    "responsible","responsibilities","achieved","achievement","managed",
    "management","developed","development","created","designed","implemented",
    "provided","supported","assisted","maintained","improved","increased",
    "reduced","delivered","deployed","tested","analyzed","reviewed","reported",
    "collaborated","coordinated","communicated","participated","performed",
    "completed","conducted","prepared","maintained","established","identified",
    "ensure","ensures","ensuring","including","included","various","multiple",
    "across","within","between","through","during","before","after","under",
    "above","based","related","required","able","available","current","previous",
    "following","including","part","full","staff","client","customer","user",
    "users","clients","business","service","services","solution","solutions",
    "process","processes","project","projects","system","systems","data","team",
    "teams","product","products","application","applications","technology",
    "technologies","platform","platforms","environment","environments","level",
    "levels","skills","tools","frameworks","languages","software","hardware",
    "network","security","quality","performance","efficiency","productivity",
    "professional","technical","analytical","communication","leadership",
    "problem","solving","teamwork","collaborative","innovative","creative",
    "detail","oriented","organized","motivated","results","driven","strong",
    "excellent","good","great","best","top","highly","key","core","main",
    "primary","secondary","additional","various","broad","deep","solid",
    "proven","demonstrated","hands","experience","expertise","proficiency",
    "familiar","knowledge","understanding","ability","capability","capacity",
    "cross","functional","end","end","result","impact","success","value",
    # ── contact / profile noise words from PDF extraction ─────────────────
    "responsible","coordination","monitoring","evaluation",
    "documentation","presentation","implementation","assessment",
    "administration","operations","execution","oversight",
    # generic verbs/adjectives that aren't real skills
    "enhance","enhanced","features","feature",
    "improve","improving","continuous","improvement",
    "creating","drive","driving","streamline",
    "ensure","ensuring","leverage","leveraging",
    "boost","boosting","innovative","outstanding",
    "loyalty","brand","utilizing","seeking",
    # city names that leak from CV addresses
    "sheffield","city","manchester","birmingham",
    "chicago","angeles","francisco","stockport",
    # template builder names
    "genius","novoresume","zety",
}


# ── Sub-specialization keyword map ────────────────────────────────────────────
# Each entry: list of keywords → score = matched / total
_SUB_SPECIALIZATIONS: dict[str, dict[str, list[str]]] = {

    "INFORMATION-TECHNOLOGY": {
        "Web Frontend Developer": [
            "react", "vue", "angular", "next.js", "nuxt", "svelte",
            "html", "css", "javascript", "typescript", "sass", "tailwind",
            "webpack", "vite", "redux", "graphql", "responsive design",
            "ui", "ux", "dom", "browser", "figma", "bootstrap",
        ],
        "Web Backend Developer": [
            "node.js", "express", "fastapi", "django", "flask", "spring boot",
            "laravel", "rails", "asp.net", "php", "rest api", "graphql",
            "microservices", "postgresql", "mysql", "mongodb", "redis",
            "rabbitmq", "kafka", "nginx", "server", "api design",
        ],
        "Full Stack Developer": [
            "full stack", "fullstack", "mern", "mean", "lamp", "next.js",
            "react", "node.js", "django", "vue", "angular", "typescript",
            "frontend", "backend", "database", "deployment", "devops",
            "rest api", "html", "css", "javascript", "sql",
        ],
        "Mobile Developer": [
            "flutter", "dart", "react native", "swift", "swiftui", "xcode",
            "kotlin", "android studio", "ios", "android", "jetpack compose",
            "objective-c", "xamarin", "ionic", "capacitor", "expo",
            "mobile app", "app store", "google play",
        ],
        "Desktop Developer": [
            "c#", "wpf", "winforms", ".net", "electron", "qt", "javafx",
            "tkinter", "pyqt", "wxpython", "desktop application",
            "windows app", "macos app", "linux gui", "winapi", "mfc",
        ],
        "Data Scientist / ML Engineer": [
            "machine learning", "deep learning", "tensorflow", "pytorch",
            "scikit-learn", "pandas", "numpy", "jupyter", "neural network",
            "nlp", "computer vision", "data science", "statistics",
            "regression", "classification", "clustering", "feature engineering",
            "model training", "model deployment", "mlops", "hugging face",
        ],
        "DevOps / Cloud Engineer": [
            "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
            "ansible", "jenkins", "ci/cd", "github actions", "helm",
            "linux", "bash", "infrastructure", "cloud", "monitoring",
            "prometheus", "grafana", "elk stack", "nginx", "load balancer",
        ],
        "Cybersecurity Analyst": [
            "penetration testing", "pentesting", "owasp", "kali linux",
            "ethical hacking", "vulnerability", "firewall", "encryption",
            "siem", "soc", "threat", "malware", "ctf", "burp suite",
            "network security", "iso 27001", "ceh", "cissp", "nmap",
        ],
        "Database Administrator": [
            "postgresql", "mysql", "oracle", "sql server", "mongodb",
            "cassandra", "redis", "dba", "database design", "normalization",
            "stored procedure", "indexing", "replication", "backup",
            "query optimization", "etl", "data warehouse", "snowflake",
        ],
    },

    "ENGINEERING": {
        "Mechanical Engineer": [
            "autocad", "solidworks", "catia", "ansys", "mechanical design",
            "thermodynamics", "fluid mechanics", "manufacturing", "cnc",
            "fea", "cfd", "gd&t", "3d printing", "hvac", "iso 9001",
        ],
        "Electrical Engineer": [
            "circuit design", "pcb", "embedded systems", "plc", "scada",
            "matlab", "simulink", "power systems", "control systems",
            "arduino", "raspberry pi", "voltage", "current", "transformer",
        ],
        "Civil Engineer": [
            "structural design", "autocad civil", "staad", "revit",
            "concrete", "steel", "surveying", "geotechnical", "highway",
            "construction management", "bim", "site supervision",
        ],
        "Software Engineer": [
            "software development", "algorithms", "data structures",
            "system design", "object oriented", "design patterns",
            "agile", "scrum", "code review", "testing", "debugging",
        ],
        "Chemical Engineer": [
            "process design", "chemical process", "aspen", "hysys",
            "reaction engineering", "mass transfer", "heat transfer",
            "piping", "safety engineering", "quality control",
        ],
    },

    "DESIGNER": {
        "UI/UX Designer": [
            "figma", "sketch", "user experience", "user interface",
            "wireframe", "prototype", "usability testing", "user research",
            "information architecture", "design system", "a/b testing",
        ],
        "Graphic Designer": [
            "photoshop", "illustrator", "indesign", "adobe creative",
            "typography", "color theory", "print design", "branding",
            "logo design", "vector", "layout design", "packaging",
        ],
        "Product Designer": [
            "product design", "design thinking", "user journey",
            "persona", "prototyping", "interaction design", "figma",
            "design strategy", "cross-functional", "stakeholder",
        ],
        "Motion Designer": [
            "after effects", "premiere", "animation", "motion graphics",
            "3d animation", "blender", "cinema 4d", "video editing",
            "visual effects", "storyboard",
        ],
    },

    "DIGITAL-MEDIA": {
        "SEO Specialist": [
            "seo", "search engine optimization", "google analytics",
            "keyword research", "backlink", "on-page seo", "technical seo",
            "serp", "ahrefs", "semrush", "google search console",
        ],
        "Social Media Manager": [
            "social media", "instagram", "facebook", "tiktok", "twitter",
            "content calendar", "community management",
            "paid social", "meta ads",
        ],
        "Content Creator": [
            "content creation", "copywriting", "blog", "storytelling",
            "video production", "youtube", "podcast", "creative writing",
            "editorial", "content strategy", "brand voice",
        ],
        "Digital Marketing Manager": [
            "digital marketing", "google ads", "ppc", "email marketing",
            "marketing automation", "crm", "hubspot", "mailchimp",
            "conversion rate", "funnel", "roi", "analytics",
        ],
    },

    "FINANCE": {
        "Financial Analyst": [
            "financial modeling", "excel", "dcf", "valuation",
            "financial statements", "variance analysis", "forecasting",
            "budgeting", "kpi", "dashboard", "reporting",
        ],
        "Investment Banker": [
            "m&a", "mergers", "acquisitions", "ipo", "capital markets",
            "pitch deck", "deal", "lbo", "cfa", "bloomberg",
            "private equity", "venture capital",
        ],
        "Risk Analyst": [
            "risk management", "var", "risk assessment", "credit risk",
            "market risk", "operational risk", "stress testing",
            "basel", "compliance", "regulatory",
        ],
        "Portfolio Manager": [
            "portfolio management", "asset allocation", "investment strategy",
            "equities", "fixed income", "derivatives", "hedge fund",
            "performance attribution", "rebalancing",
        ],
    },

    "HEALTHCARE": {
        "Nurse": [
            "nursing", "patient care", "clinical", "bedside manner",
            "vital signs", "medication", "emr", "ehr", "cpr",
            "rn", "lpn", "triage", "wound care",
        ],
        "Doctor / Physician": [
            "diagnosis", "treatment", "medical", "physician", "mbbs",
            "md", "surgery", "prescription", "clinical examination",
            "medical records", "patient history", "differential diagnosis",
        ],
        "Medical Researcher": [
            "clinical trial", "research protocol", "irb", "lab",
            "scientific writing", "data analysis", "epidemiology",
            "statistics", "hypothesis", "publication", "grant",
        ],
        "Pharmacist": [
            "pharmacy", "dispensing", "drug interaction", "pharmaceutical",
            "medication therapy", "compounding", "formulary",
            "clinical pharmacy", "prescription review",
        ],
    },

    "BANKING": {
        "Retail Banker": [
            "retail banking", "customer service", "account management",
            "loans", "deposits", "cross-selling", "teller", "branch",
        ],
        "Credit Analyst": [
            "credit analysis", "credit scoring", "loan underwriting",
            "financial statements", "debt", "collateral", "credit report",
        ],
        "Compliance Officer": [
            "compliance", "aml", "kyc", "regulatory", "audit",
            "finra", "sec", "policy", "investigation", "due diligence",
        ],
    },

    "SALES": {
        "Account Executive": [
            "account executive", "quota", "pipeline", "closing deals",
            "cold calling", "prospecting", "salesforce", "crm",
            "b2b", "b2c", "revenue", "upselling",
        ],
        "Business Development": [
            "business development", "partnerships", "market expansion",
            "lead generation", "strategic alliances", "negotiation",
            "contract", "new markets", "growth",
        ],
        "Sales Manager": [
            "sales management", "team management", "target", "kpi",
            "coaching", "performance", "territory", "forecast",
            "incentive", "go-to-market",
        ],
    },

    "HR": {
        "Talent Acquisition Specialist": [
            "recruitment", "sourcing", "headhunting", "job posting",
            "interviews", "ats", "linkedin recruiter", "onboarding",
            "employer branding", "candidate experience",
        ],
        "HR Business Partner": [
            "hrbp", "hr business partner", "performance management",
            "employee relations", "organizational development",
            "workforce planning", "succession planning", "change management",
        ],
        "Compensation & Benefits": [
            "compensation", "benefits", "payroll", "salary benchmarking",
            "total rewards", "equity", "bonus", "hris", "workday",
        ],
    },

    "TEACHER": {
        "K-12 Teacher": [
            "classroom management", "lesson planning", "curriculum",
            "k-12", "elementary", "middle school", "high school",
            "differentiated instruction", "ib", "iep",
        ],
        "University Lecturer": [
            "higher education", "university", "course design", "syllabus",
            "research", "lecture", "academic", "thesis supervision",
            "publication", "grading",
        ],
        "Online Educator": [
            "e-learning", "online course", "lms", "moodle", "udemy",
            "zoom teaching", "virtual classroom", "instructional design",
            "video production", "content creation",
        ],
    },

    "ACCOUNTANT": {
        "Tax Accountant": [
            "tax preparation", "tax planning", "irs", "corporate tax",
            "individual tax", "vat", "tax return", "tax compliance",
            "deductions", "cpa",
        ],
        "Auditor": [
            "auditing", "internal audit", "external audit", "sox",
            "gaap", "ifrs", "audit planning", "risk assessment",
            "audit report", "cia", "cpa",
        ],
        "Financial Accountant": [
            "financial reporting", "general ledger", "accounts payable",
            "accounts receivable", "reconciliation", "balance sheet",
            "income statement", "quickbooks", "sap", "month-end close",
        ],
    },

    "CONSULTANT": {
        "Management Consultant": [
            "management consulting", "strategy", "mckinsey", "bcg",
            "bain", "case study", "business analysis", "change management",
            "process improvement", "stakeholder management",
        ],
        "IT Consultant": [
            "it consulting", "digital transformation", "erp", "sap",
            "salesforce", "implementation", "project management",
            "requirements gathering", "solution architecture",
        ],
        "Financial Consultant": [
            "financial advisory", "wealth management", "financial planning",
            "investment advice", "portfolio", "retirement planning",
            "estate planning", "cfp",
        ],
    },

    "CHEF": {
        "Executive Chef": [
            "executive chef", "kitchen management", "menu development",
            "food cost", "team management", "fine dining", "michelin",
        ],
        "Pastry Chef": [
            "pastry", "baking", "dessert", "chocolate", "cake decorating",
            "bread making", "confectionery", "patisserie",
        ],
        "Line Cook": [
            "line cook", "prep cook", "grill", "saute", "mise en place",
            "food safety", "haccp", "kitchen", "recipe", "portioning",
        ],
    },

    "AVIATION": {
        "Commercial Pilot": [
            "atpl", "cpl", "ifr", "vfr", "instrument rating",
            "multi-engine", "b737", "a320", "flight hours", "type rating",
        ],
        "Flight Dispatcher": [
            "flight dispatch", "flight planning", "weather analysis",
            "notam", "fuel calculation", "ops center", "faa",
        ],
        "Aviation Maintenance": [
            "aircraft maintenance", "airframe", "powerplant", "a&p",
            "avionics", "troubleshooting", "inspection", "faa part 145",
        ],
    },

    "AGRICULTURE": {
        "Agronomist": [
            "agronomy", "soil science", "crop management", "fertilization",
            "irrigation", "plant pathology", "seed technology", "gis",
        ],
        "Agricultural Engineer": [
            "agricultural engineering", "farm machinery", "irrigation system",
            "precision agriculture", "drone", "iot sensors", "automation",
        ],
        "Farm Manager": [
            "farm management", "livestock", "harvest", "supply chain",
            "pest control", "organic farming", "certification",
        ],
    },
}


# ── Sector (specialization group) map ─────────────────────────────────────────
SECTOR_MAP: dict[str, dict] = {
    "INFORMATION-TECHNOLOGY": {"sector": "Technology",      "color": "#6e7fff", "icon": "🖥️"},
    "ENGINEERING":            {"sector": "Technology",      "color": "#6e7fff", "icon": "🖥️"},
    "DESIGNER":               {"sector": "Technology",      "color": "#6e7fff", "icon": "🖥️"},
    "DIGITAL-MEDIA":          {"sector": "Technology",      "color": "#6e7fff", "icon": "🖥️"},
    "FINANCE":                {"sector": "Business",        "color": "#a78bfa", "icon": "💼"},
    "BANKING":                {"sector": "Business",        "color": "#a78bfa", "icon": "💼"},
    "ACCOUNTANT":             {"sector": "Business",        "color": "#a78bfa", "icon": "💼"},
    "SALES":                  {"sector": "Business",        "color": "#a78bfa", "icon": "💼"},
    "CONSULTANT":             {"sector": "Business",        "color": "#a78bfa", "icon": "💼"},
    "HR":                     {"sector": "People & Org",    "color": "#2dd4bf", "icon": "👥"},
    "TEACHER":                {"sector": "People & Org",    "color": "#2dd4bf", "icon": "👥"},
    "HEALTHCARE":             {"sector": "Healthcare",      "color": "#34d399", "icon": "🏥"},
    "AGRICULTURE":            {"sector": "Healthcare",      "color": "#34d399", "icon": "🏥"},
    "CHEF":                   {"sector": "Operations",      "color": "#fbbf24", "icon": "⚙️"},
    "AVIATION":               {"sector": "Operations",      "color": "#fbbf24", "icon": "⚙️"},
}


# ── Role display names ─────────────────────────────────────────────────────────
ROLE_DISPLAY: dict[str, str] = {
    "INFORMATION-TECHNOLOGY": "Information Technology",
    "ENGINEERING":            "Engineering",
    "DESIGNER":               "Designer",
    "DIGITAL-MEDIA":          "Digital Media",
    "FINANCE":                "Finance",
    "BANKING":                "Banking",
    "ACCOUNTANT":             "Accountant",
    "SALES":                  "Sales",
    "CONSULTANT":             "Consultant",
    "HR":                     "Human Resources",
    "TEACHER":                "Teacher / Educator",
    "HEALTHCARE":             "Healthcare",
    "AGRICULTURE":            "Agriculture",
    "CHEF":                   "Chef / Culinary Arts",
    "AVIATION":               "Aviation",
}


# ── Broad role keyword tips ────────────────────────────────────────────────────
_ROLE_KEYWORDS: dict[str, list[str]] = {
    "INFORMATION-TECHNOLOGY": [
        "python", "java", "sql", "api", "git", "docker",
        "machine learning", "cloud", "agile", "database",
    ],
    "HR": [
        "recruitment", "onboarding", "payroll", "hris",
        "performance management", "employee relations", "training",
    ],
    "FINANCE": [
        "financial modeling", "excel", "accounting", "audit",
        "budget", "forecasting", "cfa", "valuation", "tax",
    ],
    "HEALTHCARE": [
        "patient care", "clinical", "ehr", "nursing",
        "diagnosis", "medical records", "cpr", "hipaa",
    ],
    "CHEF": [
        "culinary", "food safety", "kitchen management",
        "menu planning", "haccp", "catering", "pastry",
    ],
    "ENGINEERING": [
        "autocad", "solidworks", "project management",
        "quality control", "mechanical", "electrical", "cad",
    ],
    "SALES": [
        "crm", "negotiation", "lead generation", "revenue",
        "cold calling", "pipeline", "salesforce", "quota",
    ],
    "BANKING": [
        "risk management", "compliance", "aml", "kyc",
        "credit analysis", "investment", "portfolio", "basel",
    ],
    "ACCOUNTANT": [
        "gaap", "quickbooks", "tax preparation", "audit",
        "accounts payable", "accounts receivable", "reconciliation",
    ],
    "TEACHER": [
        "curriculum", "lesson planning", "classroom management",
        "assessment", "differentiated instruction", "ib", "stem",
    ],
    "DESIGNER": [
        "photoshop", "illustrator", "figma", "ui/ux",
        "typography", "branding", "wireframe", "adobe",
    ],
    "DIGITAL-MEDIA": [
        "seo", "social media", "content creation", "analytics",
        "google ads", "facebook ads", "copywriting", "campaign",
    ],
    "CONSULTANT": [
        "stakeholder management", "strategy", "analysis",
        "presentation", "problem solving", "client management",
    ],
    "AVIATION": [
        "atpl", "cpl", "ifr", "safety management",
        "aircraft", "navigation", "flight operations",
    ],
    "AGRICULTURE": [
        "crop management", "irrigation", "soil science",
        "agronomy", "pest control", "harvest", "gis",
    ],
}


# ── Text utilities ─────────────────────────────────────────────────────────────

# Technical terms protection (Matches train_model_v2.py)
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

_SKILL_TERM_EXCLUDE = {
    "r", "go", "it", "hr", "cv", "pm", "ba", "vp", "db", "os", "vm",
    "frontend", "backend", "server", "team management", "performance",
    # contact / profile info — never a skill
    "linkedin", "github", "portfolio", "website", "twitter", "facebook",
    # generic marketing / HR terms too broad to be meaningful skills
    "engagement", "community", "influencer", "interviews", "onboarding",
    "presentations", "reporting", "training", "testing", "monitoring",
    # education / institution words — not skills
    "university", "college", "institute", "school", "bachelor", "master",
    "degree", "diploma", "certification", "graduate", "undergraduate",
}


def _build_skill_vocabulary() -> list[str]:
    terms: set[str] = set()
    for keywords in _ROLE_KEYWORDS.values():
        terms.update(k.lower() for k in keywords)
    for role_specs in _SUB_SPECIALIZATIONS.values():
        for keywords in role_specs.values():
            terms.update(k.lower() for k in keywords)
    terms.update(
        {
            "aws", "azure", "gcp", "api", "rest api", "api design", "sql",
            "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
            "git", "github actions", "ci/cd", "python", "java", "javascript",
            "typescript", "react", "angular", "vue", "fastapi", "django",
            "flask", "automated testing", "unit testing", "production monitoring",
            "performance optimization", "c#", "c++", ".net", "asp.net",
        }
    )
    cleaned_terms = {
        t.strip().lower()
        for t in terms
        if t and t.strip().lower() not in _SKILL_TERM_EXCLUDE
    }
    return sorted(cleaned_terms, key=lambda t: (-len(t), t))


_SKILL_VOCABULARY = _build_skill_vocabulary()


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=512)
def _term_pattern(term: str) -> str:
    parts: list[str] = []
    for part in re.split(r"[\s/-]+", term.lower()):
        if not part:
            continue
        parts.append(r"apis?" if part == "api" else re.escape(part))
    if not parts:
        return ""
    # Allow one or more whitespace characters / hyphens between words
    # to handle PDFs that embed extra spaces between characters.
    separator = r"[\s/-]+" if len(parts) > 1 else ""
    pattern = separator.join(parts)
    return rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])"



def _term_match_position(term: str, text: str) -> int | None:
    pattern = _term_pattern(term)
    match = re.search(pattern, text.lower()) if pattern else None
    return match.start() if match else None


def _has_skill_term(term: str, text: str) -> bool:
    return _term_match_position(term, text) is not None

def clean_text(text: str) -> str:
    """Improved cleaning that preserves technical abbreviations. Matches v2 training."""
    if not isinstance(text, str):
        return ""

    # Basic cleaning
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
    text = text.lower()

    # Protect key terms
    protected_map = {}
    for term in _PROTECTED_TERMS:
        safe_key = f"PROT_{term.replace('#','SHARP').replace('+','PLUS')}__"
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(term) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, text):
            protected_map[safe_key] = term
            text = re.sub(pattern, f" {safe_key} ", text)

    # Remove noise
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\b(?!PROT_)\w{1,2}\b", " ", text)

    # Restore
    for safe_key, original in protected_map.items():
        text = text.replace(safe_key, original)

    return re.sub(r"\s+", " ", text).strip()


def _extract_ats_terms(text: str) -> set[str]:
    """
    Robust ATS term extraction with three complementary strategies:
    1) Direct multi-word vocabulary matching (most precise for skills)
    2) spaCy token extraction (good for single-word technical terms)
    3) Regex fallback (prevents empty-skill false positives)
    """
    if not text:
        return set()

    terms: set[str] = set()

    # 1) Multi-word vocabulary matching — catches "financial modeling",
    #    "machine learning", etc. that spaCy splits into individual tokens.
    for vocab_term in _SKILL_VOCABULARY:
        if _has_skill_term(vocab_term, text):
            terms.add(vocab_term)

    cleaned = clean_text(text)
    if not cleaned:
        return terms

    # 2) spaCy extraction — good for individual technical tokens
    if _nlp:
        try:
            doc = _nlp(cleaned)
            for token in doc:
                t = token.text.lower().strip()
                if (
                    t
                    and len(t) > 2
                    and t not in _NOISE_WORDS
                    and (
                        token.pos_ in ("PROPN", "NOUN", "ADJ", "VERB")
                        or t in _PROTECTED_TERMS
                    )
                ):
                    terms.add(t)
        except Exception:
            pass

    # 3) Regex fallback — prevent empty-result edge cases
    if len(terms) < 8:
        regex_terms = re.findall(r"[a-z][a-z0-9+#./-]{2,}", cleaned.lower())
        for t in regex_terms:
            if t in _NOISE_WORDS:
                continue
            terms.add(t)

    # Remove any noise that slipped through
    terms = {t for t in terms if t not in _NOISE_WORDS}
    return terms


def extract_smart_skills(text: str, predicted_role: str | None = None) -> list[str]:
    """
    AI-driven skill extraction using the trained LinearSVC's TF-IDF feature weights.

    Strategy:
    1. Transform the CV text through the trained TF-IDF (word) vectorizer.
    2. Look up the LinearSVC decision weights for the predicted role class.
    3. For every word present in the CV, rank it by (tfidf_score × model_weight).
    4. Filter out noise/stop words and return the top-ranked terms as skills.

    Falls back to the vocabulary-based method if the model is not loaded yet.
    """
    if not text:
        return []

    # ── AI-driven path ────────────────────────────────────────────────────────
    if model is not None and tfidf_word is not None and encoder is not None:
        try:
            cleaned = clean_text(text)
            # TF-IDF vector for this CV
            cv_vec = tfidf_word.transform([cleaned])   # shape (1, n_features)

            # Determine which class index to use for weights
            role_key = (predicted_role or "").upper().replace(" ", "-")
            classes = list(encoder.classes_)
            if role_key in classes:
                cls_idx = classes.index(role_key)
            else:
                # Fall back to the class with highest decision score
                cls_idx = int(model.decision_function(cv_vec)[0].argmax())

            # Feature weights for this class (LinearSVC coef_ shape: n_classes × n_features)
            weights = model.coef_[cls_idx]            # shape (n_features,)

            # Score = tfidf_value × class_weight  (only for non-zero CV terms)
            cv_array   = cv_vec.toarray()[0]
            vocab      = tfidf_word.get_feature_names_out()  # feature name per column
            scores     = cv_array * weights                   # element-wise

            # Collect (score, term) pairs where CV actually contains the term
            candidates: list[tuple[float, str]] = []
            for idx in scores.argsort()[::-1]:
                if cv_array[idx] == 0:
                    continue                        # term not in CV
                term  = vocab[idx]
                score = float(scores[idx])
                if score <= 0:
                    break                          # remaining weights are negative
                # Filter noise
                if (
                    term in _SKILL_TERM_EXCLUDE
                    or term in _NOISE_WORDS
                    or len(term) <= 2
                    or term.isdigit()
                ):
                    continue
                candidates.append((score, term))
                if len(candidates) == 20:
                    break

            if candidates:
                return [term for _, term in candidates[:15]]
        except Exception:
            pass   # fall through to vocabulary method on any error

    # ── Fallback: vocabulary-based (used before model loads) ─────────────────
    matches: list[tuple[int, str]] = []
    for term in _SKILL_VOCABULARY:
        pos = _term_match_position(term, text)
        if pos is not None:
            matches.append((pos, term))

    matches.sort(key=lambda item: (item[0], -len(item[1])))
    skills: list[str] = []
    occupied: list[tuple[int, int]] = []
    for pos, term in matches:
        end = pos + len(term)
        if any(pos >= start and end <= stop for start, stop in occupied):
            continue
        if any(term != existing and term in existing for existing in skills):
            continue
        if term in _SKILL_TERM_EXCLUDE or term in _NOISE_WORDS:
            continue
        skills.append(term)
        occupied.append((pos, end))

    return skills



def detect_sub_specialization(role: str, cv_text: str) -> dict:
    """
    Detects which sub-specialization within a role best matches the CV.
    Returns ranked list of sub-specs with match scores, plus the top match.

    Example for INFORMATION-TECHNOLOGY:
      top: "Web Frontend Developer"
      scores: [("Web Frontend Developer", 0.42), ("Full Stack Developer", 0.31), ...]
    """
    role_key = role.upper().replace(" ", "-")
    sub_specs = _SUB_SPECIALIZATIONS.get(role_key, {})

    if not sub_specs:
        return {"top": None, "scores": []}

    cv_lower   = cv_text.lower()
    raw_scores: list[tuple[str, float]] = []

    for spec_name, keywords in sub_specs.items():
        matched = sum(1 for kw in keywords if kw in cv_lower)
        score   = round(matched / len(keywords) * 100, 1) if keywords else 0.0
        raw_scores.append((spec_name, score))

    ranked = sorted(raw_scores, key=lambda x: x[1], reverse=True)
    top    = ranked[0][0] if ranked[0][1] > 0 else None

    return {
        "top":    top,
        "scores": [{"name": n, "score": s} for n, s in ranked if s > 0],
    }


def detect_career_level(cv_text: str) -> str:
    """
    Detects career seniority from the CV text based on years-of-experience
    patterns and seniority keywords.
    """
    text = cv_text.lower()

    # Years of experience pattern: "5 years", "10+ years", etc.
    yoe_matches = re.findall(
        r'(\d+)\s*\+?\s*years?\s+of\s+experience|(\d+)\s*\+?\s*years?\s+experience',
        text, re.IGNORECASE
    )
    years = [int(m[0] or m[1]) for m in yoe_matches if m[0] or m[1]]
    max_years = max(years) if years else 0

    if re.search(r'\b(cto|ceo|vp|vice president|director|head of|principal|staff engineer)\b', text):
        return "Lead / Executive"
    if max_years >= 8 or re.search(r'\b(senior|sr\.|lead|architect|expert|seasoned)\b', text):
        return "Senior"
    if max_years >= 3 or re.search(r'\b(mid.level|intermediate|experienced)\b', text):
        return "Mid-level"
    if max_years >= 1 or re.search(r'\b(junior|jr\.|associate|entry.level|graduate|fresher|intern)\b', text):
        return "Junior"
    return "Not Specified"


# ── Default JD templates (used when user provides no job description) ─────────
_DEFAULT_JD: dict[str, str] = {
    "INFORMATION-TECHNOLOGY": (
        "Software engineer with experience in Python, Java, or JavaScript. "
        "Proficient in REST APIs, databases, cloud platforms (AWS/Azure/GCP), "
        "Docker, Git, agile development, and system design."
    ),
    "ENGINEERING": (
        "Mechanical or electrical engineer with AutoCAD, SolidWorks experience. "
        "Project management, quality control, technical documentation, and "
        "hands-on manufacturing or construction site experience."
    ),
    "HR": (
        "HR professional skilled in recruitment, onboarding, payroll, HRIS, "
        "performance management, employee relations, and talent acquisition."
    ),
    "FINANCE": (
        "Finance professional with financial modeling, Excel, forecasting, "
        "budgeting, accounting, audit, and reporting experience."
    ),
    "DESIGNER": (
        "Designer proficient in Figma, Adobe Photoshop, Illustrator. "
        "UI/UX design, wireframing, prototyping, branding, and typography."
    ),
    "DIGITAL-MEDIA": (
        "Digital marketing professional with SEO, Google Ads, social media, "
        "content creation, analytics, and campaign management experience."
    ),
    "SALES": (
        "Sales professional with CRM tools, lead generation, B2B sales, "
        "negotiation, pipeline management, and quota achievement experience."
    ),
    "BANKING": (
        "Banking professional with risk management, compliance, AML, KYC, "
        "credit analysis, and financial product knowledge."
    ),
    "HEALTHCARE": (
        "Healthcare professional with patient care, clinical skills, EMR/EHR, "
        "medical documentation, and compliance experience."
    ),
    "ACCOUNTANT": (
        "Accountant with GAAP, QuickBooks, tax preparation, audit, "
        "accounts payable/receivable, and month-end close experience."
    ),
    "TEACHER": (
        "Educator with curriculum design, lesson planning, classroom management, "
        "student assessment, and differentiated instruction experience."
    ),
    "CONSULTANT": (
        "Consultant with strategy, stakeholder management, business analysis, "
        "process improvement, and client presentation experience."
    ),
    "CHEF": (
        "Chef with kitchen management, menu development, food safety (HACCP), "
        "team leadership, and culinary expertise."
    ),
    "AVIATION": (
        "Aviation professional with flight operations, safety management, "
        "aircraft systems, and regulatory compliance experience."
    ),
    "AGRICULTURE": (
        "Agricultural professional with crop management, soil science, "
        "irrigation, agronomy, and farm operations experience."
    ),
}


def compute_ats_score(
    cv_text: str,
    target_jd: str | None = None,
    predicted_role: str = "",
    cv_embedding=None,
    return_breakdown: bool = False
) -> float | tuple[float, dict]:
    """
    Cosine-similarity ATS score (0-100) via sentence-transformers.
    - If target_jd is provided: compares CV against that JD.
    - If target_jd is empty: falls back to a role-based default JD template.
    - Returns -1.0 only if the model is unavailable.
    """
    jd = (target_jd or "").strip()
    if not jd:
        # Use the default JD for the predicted role, or a generic fallback
        jd = _DEFAULT_JD.get(
            predicted_role.upper().replace(" ", "-"),
            "Professional with relevant skills, experience, and industry knowledge."
        )

    # 1. Keyword alignment score (coverage + precision)
    cv_terms = _extract_ats_terms(cv_text)
    jd_terms = _extract_ats_terms(jd)

    overlap_terms = cv_terms.intersection(jd_terms)
    if jd_terms:
        # Strict mathematical calculation
        coverage = len(overlap_terms) / max(len(jd_terms), 1)           # How much JD is covered
        precision = len(overlap_terms) / max(len(cv_terms), 1)          # How focused CV is
        
        # Heavy weight on covering the required JD terms (85%)
        keyword_score = (coverage * 0.85 + precision * 0.15) * 100.0
    else:
        # No JD terms found = no keyword match
        keyword_score = 0.0

    # 2. Semantic Score
    semantic_score = 0.0
    if _similarity_model:
        jd_for_sem = clean_text(jd)[:2500]
        if jd_for_sem:
            if cv_embedding is not None:
                emb_cv = cv_embedding
            else:
                cv_for_sem = clean_text(cv_text)[:2500]
                emb_cv = _similarity_model.encode(cv_for_sem, convert_to_tensor=True)
            
            emb_jd = _similarity_model.encode(jd_for_sem, convert_to_tensor=True)
            sim = st_util.pytorch_cos_sim(emb_cv, emb_jd).item()
            
            # Realistic strict calibration: Requires sim > 0.10 to start scoring
            semantic_score = max(0.0, min(100.0, (sim - 0.10) / 0.60 * 100.0))
        else:
            semantic_score = keyword_score
    else:
        semantic_score = keyword_score

    # Final blend: 60% Keyword Requirements, 40% Semantic Meaning
    final_score = (keyword_score * 0.60) + (semantic_score * 0.40)

    # Mild penalty for very short CVs to avoid unrealistically high matches.
    if len(cv_text.split()) < 80:
        final_score *= 0.97

    final_score = max(0.0, min(100.0, final_score))
    final_val = round(final_score, 1)

    if return_breakdown:
        # Skills match calculation
        user_skills = extract_smart_skills(cv_text, predicted_role=predicted_role)
        if target_jd:
            jd_skills = extract_smart_skills(target_jd)
            missing_skills = [
                skill for skill in jd_skills if not _has_skill_term(skill, cv_text)
            ][:10]
        else:
            role_key = predicted_role.upper().replace(" ", "-")
            role_kws = _ROLE_KEYWORDS.get(role_key, [])
            missing_skills = [kw for kw in role_kws if not _has_skill_term(kw, cv_text)][:10]

        skills_match_score = len(user_skills) / max(len(user_skills) + len(missing_skills), 1) * 100.0

        # Resume quality score calculation
        wc = len(cv_text.split())
        quality_score = 100.0
        if wc < 150:
            quality_score -= 20
        elif wc < 250:
            quality_score -= 10
        elif wc > 1000:
            quality_score -= 10
        elif wc > 1500:
            quality_score -= 20

        import re
        numbers_found = re.findall(
            r'\d+%|\$[\d,]+|\d+\s*(?:users|clients|projects|employees|years)', cv_text, re.IGNORECASE
        )
        if len(numbers_found) == 0:
            quality_score -= 25
        elif len(numbers_found) == 1:
            quality_score -= 15
        elif len(numbers_found) == 2:
            quality_score -= 5

        cv_lower = cv_text.lower()
        if 'education' not in cv_lower:
            quality_score -= 15
        if not any(k in cv_lower for k in ['experience', 'work', 'history', 'employment']):
            quality_score -= 15
        if 'project' not in cv_lower:
            quality_score -= 10

        if not re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text):
            quality_score -= 10
        if not re.search(r'\+?\d[\d\-\(\)\s]{7,}\d', cv_text):
            quality_score -= 10

        quality_score = max(0.0, min(100.0, quality_score))

        calc_desc = "ATS Score = (Keyword Match Score * 60%) + (Semantic Similarity Score * 40%)"
        if len(cv_text.split()) < 80:
            calc_desc += " (with a 3% penalty for low word count)"

        breakdown = {
            "keyword_match_score": round(keyword_score, 1),
            "skills_match_score": round(skills_match_score, 1),
            "semantic_similarity_score": round(semantic_score, 1),
            "resume_quality_score": round(quality_score, 1),
            "calculation_description": calc_desc
        }
        return final_val, breakdown

    return final_val


# ── Tips generation ────────────────────────────────────────────────────────────

def generate_tips(
    predicted_role:  str,
    cv_text:         str,
    ats_score:       float       = -1.0,
    missing_skills:  list[str] | None = None,
    user_skills:     list[str] | None = None,
    sub_spec:        str | None = None,
) -> dict:
    if missing_skills is None:
        missing_skills = []
    if user_skills is None:
        user_skills = []

    tips:      list[dict] = []
    cv_lower:  str        = cv_text.lower()
    role_key:  str        = predicted_role.upper().replace(" ", "-")
    keywords:  list[str]  = _ROLE_KEYWORDS.get(role_key, [])

    # 1. ATS score tip
    if ats_score >= 0:
        if ats_score >= 80:
            ats_msg = "Excellent! Your CV is highly optimised for ATS scanners."
        elif ats_score >= 50:
            ats_msg = "Good effort! Add more specific keywords from the job description to improve matching."
        else:
            ats_msg = "Low match score. Simplify your layout and incorporate more industry-specific terms."
        tips.append({"type": "format", "title": "ATS Compatibility", "message": ats_msg})

    # 2. Sub-specialization tip
    if sub_spec:
        sub_kws = _SUB_SPECIALIZATIONS.get(role_key, {}).get(sub_spec, [])
        sub_missing = [kw for kw in sub_kws if kw not in cv_lower][:3]
        if sub_missing:
            tips.append({
                "type":    "keywords",
                "title":   f"Strengthen Your {sub_spec} Profile",
                "message": f"Add these keywords to reinforce your specialization: {', '.join(sub_missing)}.",
            })

    # 3. Missing broad keywords tip
    keyword_gaps = [kw for kw in keywords if kw not in cv_lower]
    if keyword_gaps:
        tips.append({
            "type":    "keywords",
            "title":   "Add Missing Keywords",
            "message": f"Consider adding: {', '.join(keyword_gaps[:5])}.",
        })

    # 4. Skill density tip
    if missing_skills:
        m_str = ", ".join(missing_skills[:2])
        skills_msg = f"Found {len(user_skills)} relevant skills. Adding '{m_str}' would boost your ranking."
    else:
        skills_msg = f"Strong skill density! AI identified {len(user_skills)} professional keywords."
    tips.append({"type": "achievements", "title": "Skill Density", "message": skills_msg})

    # 5. CV length tip
    word_count = len(cv_text.split())
    if word_count < 200:
        tips.append({
            "type":    "length",
            "title":   "CV Too Short",
            "message": "Your CV seems short. Add more details about your experience and achievements.",
        })
    elif word_count > 1000:
        tips.append({
            "type":    "length",
            "title":   "CV Too Long",
            "message": "Consider trimming your CV to 1-2 pages. Focus on the most relevant experience.",
        })

    # 6. Measurable achievements tip
    numbers_found = re.findall(
        r'\d+%|\$[\d,]+|\d+\s*(?:users|clients|projects|employees|years)', cv_text, re.IGNORECASE
    )
    if len(numbers_found) < 2:
        tips.append({
            "type":    "achievements",
            "title":   "Add Measurable Impact",
            "message": "Use numbers to show impact, e.g. 'Improved performance by 30%' or 'Managed a team of 10'.",
        })

    # 7. Always-present tailoring tip
    role_display = ROLE_DISPLAY.get(role_key, predicted_role)
    tips.append({
        "type":    "format",
        "title":   "Tailor Your CV",
        "message": f"Customise your CV for each {role_display} job application.",
    })

    return {"tips": tips, "total": len(tips)}


# ── Main predict function ──────────────────────────────────────────────────────

def predict(cv_text: str, target_jd: str = "") -> dict:
    """
    Full CV analysis:
      - Role classification (24 categories)
      - Sub-specialization detection (Web Frontend, Mobile, Desktop, etc.)
      - Sector/industry grouping
      - Career level detection
      - ATS semantic scoring
      - Smart skill extraction (noise-filtered)
      - Improvement tips
    """
    cleaned = clean_text(cv_text)
    if not cleaned.strip():
        raise ValueError("CV text is empty after cleaning.")

    _ensure_classifier_ready()
    cv_emb_tensor = None

    if tfidf_word and tfidf_char:
        # V2 / V3 Path: Build features manually
        from scipy.sparse import hstack, csr_matrix
        
        x_w = tfidf_word.transform([cleaned])
        x_c = tfidf_char.transform([cleaned])
        similarity_model = _ensure_similarity_model()
        if similarity_model is not None:
            cv_emb_tensor = similarity_model.encode(cleaned, convert_to_tensor=True, normalize_embeddings=True)
            emb = cv_emb_tensor.cpu().numpy().reshape(1, -1)
        else:
            emb = np.zeros((1, 384))
        
        x_combined = hstack([x_w, x_c])
        
        pred_encoded = model.predict(x_combined)[0]
        probas       = _predict_proba_any(model, x_combined)[0]
    elif _is_text_pipeline(model):
        # V1 Path: Pipeline handles everything
        pred_encoded = model.predict([cleaned])[0]
        probas       = _predict_proba_any(model, [cleaned])[0]
    else:
        raise RuntimeError(
            "Classifier expects vectorized features, but TF-IDF artifacts were not loaded. "
            "Run train_model.py to regenerate saved_model/."
        )

    pred_label   = encoder.inverse_transform([pred_encoded])[0]
    confidence   = _display_confidence(probas)

    all_scores = {
        encoder.classes_[i]: round(float(probas[i] * 100), 2)
        for i in np.argsort(probas)[::-1]
    }

    # Conservative post-processing rules fix common near-tie misclassifications.
    if os.getenv("APPLY_POSTPROCESS", "1") != "0":
        try:
            from backend.services.postprocess import adjust_predicted_role

            adjusted = adjust_predicted_role(cv_text=cv_text, predicted_role=pred_label, all_scores=all_scores)
            if adjusted != pred_label:
                pred_label = adjusted
                confidence = max(confidence, 55.0)
        except Exception:
            pass

    # Sector info
    sector_info   = SECTOR_MAP.get(pred_label, {"sector": "General", "color": "#7b82a8", "icon": "💼"})
    role_display  = ROLE_DISPLAY.get(pred_label, pred_label.replace("-", " ").title())

    # Sub-specialization detection
    sub_spec_result = detect_sub_specialization(pred_label, cv_text)

    # Career level
    career_level = detect_career_level(cv_text)

    # ATS score (pass predicted role so default JD fallback is role-aware)
    ats_score, ats_breakdown = compute_ats_score(
        cv_text, target_jd, predicted_role=pred_label, cv_embedding=cv_emb_tensor, return_breakdown=True
    )

    # Related roles in same sector
    related_roles = [
        {"role": role, "display": ROLE_DISPLAY.get(role, role), "emoji": _role_emoji(role)}
        for role, info in SECTOR_MAP.items()
        if info["sector"] == sector_info["sector"] and role != pred_label
    ]

    # Skill extraction & gap analysis
    user_skills       = extract_smart_skills(cv_text, predicted_role=pred_label)
    
    has_target_jd = len(target_jd.split()) >= 20

    if target_jd:
        jd_skills = extract_smart_skills(target_jd)
        missing_skills = [
            skill
            for skill in jd_skills
            if not _has_skill_term(skill, cv_text)
        ][:10]
    else:
        role_key       = pred_label.upper().replace(" ", "-")
        role_kws       = _ROLE_KEYWORDS.get(role_key, [])
        missing_skills = [kw for kw in role_kws if not _has_skill_term(kw, cv_text)][:10]
        
    # Show a hard mismatch warning only when the JD signal is strong and the gap is severe.
    is_mismatch = bool(
        has_target_jd
        and ats_score >= 0
        and ats_score < 25
        and len(missing_skills) >= 6
    )

    # Tips
    tips = generate_tips(
        predicted_role = pred_label,
        cv_text        = cv_text,
        ats_score      = ats_score,
        missing_skills = missing_skills,
        user_skills    = user_skills,
        sub_spec       = sub_spec_result.get("top"),
    )

    # Extract terms for keyword matches and missing terms
    jd = (target_jd or "").strip()
    if not jd:
        jd = _DEFAULT_JD.get(
            pred_label.upper().replace(" ", "-"),
            "Professional with relevant skills, experience, and industry knowledge."
        )
    cv_terms = _extract_ats_terms(cv_text)
    jd_terms = _extract_ats_terms(jd)
    matched_keywords = sorted(list(cv_terms.intersection(jd_terms)))
    missing_keywords = sorted(list(jd_terms - cv_terms))

    # Strengths and Weaknesses
    resume_strengths = []
    resume_weaknesses = []

    # Keyword match check
    if ats_breakdown["keyword_match_score"] >= 70:
        resume_strengths.append("Good keyword coverage matching the target role")
    elif ats_breakdown["keyword_match_score"] < 50:
        resume_weaknesses.append("Low keyword match (needs more industry-specific terms)")

    # Skills density check
    if len(user_skills) >= 8:
        resume_strengths.append("Strong technical/professional skill density")
    elif len(user_skills) < 4:
        resume_weaknesses.append("Technical skills section could be expanded with more tools/skills")

    # Education check
    cv_lower = cv_text.lower()
    if 'education' in cv_lower or 'academic' in cv_lower:
        resume_strengths.append("Clear and relevant academic education section")
    else:
        resume_weaknesses.append("Missing academic education section")

    # Projects check
    if 'project' in cv_lower:
        resume_strengths.append("Project experience details are well documented")
    else:
        resume_weaknesses.append("No portfolio projects or personal projects mentioned")

    # Measurable achievements (metrics) check
    import re
    numbers_found = re.findall(
        r'\d+%|\$[\d,]+|\d+\s*(?:users|clients|projects|employees|years)', cv_text, re.IGNORECASE
    )
    if len(numbers_found) >= 3:
        resume_strengths.append("Strong use of quantified achievements and performance metrics")
    else:
        resume_weaknesses.append("Lack of quantified achievements (add numbers, percentages, or savings)")

    # Length check
    wc = len(cv_text.split())
    if 250 <= wc <= 800:
        resume_strengths.append("Optimal resume length (highly reader-friendly)")
    elif wc < 150:
        resume_weaknesses.append("Resume is too short (lacks detailed experience highlights)")
    elif wc > 1000:
        resume_weaknesses.append("Resume is too long (aim for 1 to 2 pages max)")

    # Semantic similarity check
    if ats_breakdown["semantic_similarity_score"] >= 65:
        resume_strengths.append("Strong semantic alignment with role expectations")
    elif ats_breakdown["semantic_similarity_score"] < 45:
        resume_weaknesses.append("Low semantic similarity to the job description requirements")

    # Portfolio/links check
    if re.search(r'linkedin\.com|github\.com|portfolio', cv_lower):
        resume_strengths.append("Includes links to professional networks/portfolio")
    else:
        resume_weaknesses.append("Missing links to professional networks (LinkedIn or GitHub portfolio)")

    # Fallbacks
    if len(resume_strengths) < 2:
        resume_strengths.extend([
            "Professional resume layout structure",
            "Includes essential contact information"
        ])
    if len(resume_weaknesses) == 0:
        resume_weaknesses.extend([
            "Could be enhanced with certifications related to the target role",
            "Add a robust professional summary at the top of the resume"
        ])

    resume_strengths = resume_strengths[:5]
    resume_weaknesses = resume_weaknesses[:5]

    # Recommendations
    ats_recommendations = []
    if missing_keywords:
        ats_recommendations.append(f"Incorporate missing keywords naturally: {', '.join(missing_keywords[:4])}.")
    if len(numbers_found) < 2:
        ats_recommendations.append("Incorporate measurable impact using numbers/percentages (e.g. 'boosted sales by 20%', 'managed a $50k project budget').")
    if 'project' not in cv_lower:
        ats_recommendations.append("Add a 'Projects' section highlighting personal or academic work demonstrating hands-on expertise.")
    if len(user_skills) < 6:
        ats_recommendations.append("Expand the core skills section to list more relevant technical tools, frameworks, or methodologies.")
    if not re.search(r'linkedin\.com|github\.com', cv_lower):
        ats_recommendations.append("Add links to your LinkedIn profile or GitHub/portfolio website to showcase active projects and professional history.")
    if wc < 150:
        ats_recommendations.append("Expand on your job roles by adding 3-4 bullet points describing specific achievements and responsibilities for each.")
    elif wc > 1000:
        ats_recommendations.append("Trim your resume length. Focus on the last 10 years of experience and prioritize bullet points matching the JD.")
    ats_recommendations.append(f"Tailor your CV's professional summary and work history bullet points specifically to a {role_display} position.")
    ats_recommendations = ats_recommendations[:5]

    # Enrich tips dictionary
    tips["ats_breakdown"] = ats_breakdown
    tips["matched_keywords"] = matched_keywords
    tips["missing_keywords"] = missing_keywords
    tips["ats_recommendations"] = ats_recommendations
    tips["resume_strengths"] = resume_strengths
    tips["resume_weaknesses"] = resume_weaknesses

    return {
        "predicted_role":    pred_label,
        "role_display":      role_display,
        "confidence":        round(confidence, 2),
        "sector":            sector_info["sector"],
        "sector_color":      sector_info["color"],
        "sector_icon":       sector_info["icon"],
        "sub_specialization": sub_spec_result,
        "career_level":      career_level,
        "related_roles":     related_roles,
        "ats_score":         ats_score if ats_score >= 0 else None,
        "all_scores":        all_scores,
        "extracted_skills":  user_skills[:15],
        "missing_skills":    missing_skills,
        "is_mismatch":       is_mismatch,
        "tips":              tips,
    }


def _role_emoji(role: str) -> str:
    _map = {
        "INFORMATION-TECHNOLOGY": "💻", "HR": "👥",                "FINANCE": "💰",
        "DESIGNER": "🎨",               "SALES": "📈",             "BANKING": "🏦",
        "HEALTHCARE": "🏥",             "CHEF": "👨‍🍳",              "ENGINEERING": "⚙️",
        "ACCOUNTANT": "📊",             "TEACHER": "📚",           "DIGITAL-MEDIA": "📱",
        "CONSULTANT": "🤝",             "AVIATION": "✈️",          "AGRICULTURE": "🌱",
        "BUSINESS-DEVELOPMENT": "🚀",   "ARTS": "🎭",              "ADVOCATE": "⚖️",
        "APPAREL": "👗",                "AUTOMOBILE": "🚗",        "BPO": "📞",
        "CONSTRUCTION": "🏗️",           "PUBLIC-RELATIONS": "📣",  "TOURISM": "🌍",
    }
    return _map.get(role, "💼")
