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
            "linkedin", "content calendar", "engagement", "community management",
            "influencer", "paid social", "meta ads",
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
    Robust ATS term extraction with spaCy-first and regex fallback.
    Keeps technical tokens and avoids empty-term edge cases.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return set()

    terms: set[str] = set()

    # 1) spaCy extraction when available
    if _nlp:
        try:
            doc = _nlp(cleaned)
            for token in doc:
                t = token.text.lower().strip()
                if (
                    t
                    and len(t) > 2
                    and t not in _NOISE_WORDS
                    and (token.pos_ in ("PROPN", "NOUN") or t in _PROTECTED_TERMS)
                ):
                    terms.add(t)
        except Exception:
            pass

    # 2) fallback regex extraction to prevent empty-skill false positives
    if len(terms) < 8:
        regex_terms = re.findall(r"[a-z][a-z0-9+#./-]{2,}", cleaned.lower())
        for t in regex_terms:
            if t in _NOISE_WORDS:
                continue
            terms.add(t)

    return terms


def extract_smart_skills(text: str) -> list[str]:
    """
    Uses spaCy POS tagging to extract meaningful noun/proper-noun keywords,
    filtered against a noise-word blocklist for clean results.
    """
    if not _nlp or not text:
        return []
    doc = _nlp(text.lower())
    skills = [
        token.text.strip()
        for token in doc
        if token.pos_ in ("PROPN", "NOUN")
        and len(token.text) > 2
        and token.text.lower() not in _NOISE_WORDS
        and token.is_alpha
    ]
    return list(set(skills))


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


def compute_ats_score(cv_text: str, target_jd: str, predicted_role: str = "") -> float:
    """
    Cosine-similarity ATS score (0-100) via sentence-transformers.
    - If target_jd is provided: compares CV against that JD.
    - If target_jd is empty: falls back to a role-based default JD template.
    - Returns -1.0 only if the model is unavailable.
    """
    jd = target_jd.strip()
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
        cv_for_sem = clean_text(cv_text)[:2500]
        jd_for_sem = clean_text(jd)[:2500]
        if cv_for_sem and jd_for_sem:
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
    return round(final_score, 1)


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

    if tfidf_word and tfidf_char:
        # V2 / V3 Path: Build features manually
        from scipy.sparse import hstack, csr_matrix
        
        x_w = tfidf_word.transform([cleaned])
        x_c = tfidf_char.transform([cleaned])
        similarity_model = _ensure_similarity_model()
        emb = similarity_model.encode([cleaned], normalize_embeddings=True)
        
        if scaler is not None:
            x_e = csr_matrix(scaler.transform(emb))
        else:
            x_e = csr_matrix(emb)
            
        x_combined = hstack([x_w, x_c, x_e])
        
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

    # Optional post-processing rules (disabled by default).
    if os.getenv("APPLY_POSTPROCESS", "0") == "1":
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
    ats_score = compute_ats_score(cv_text, target_jd, predicted_role=pred_label)

    # Related roles in same sector
    related_roles = [
        {"role": role, "display": ROLE_DISPLAY.get(role, role), "emoji": _role_emoji(role)}
        for role, info in SECTOR_MAP.items()
        if info["sector"] == sector_info["sector"] and role != pred_label
    ]

    # Skill extraction & gap analysis
    user_skills       = extract_smart_skills(cv_text)
    user_skills_lower = {s.lower() for s in user_skills}
    
    has_target_jd = len(target_jd.split()) >= 20

    if target_jd:
        jd_skills      = set(extract_smart_skills(target_jd))
        missing_skills = sorted([s for s in jd_skills if s.lower() not in user_skills_lower])[:10]
    else:
        role_key       = pred_label.upper().replace(" ", "-")
        role_kws       = _ROLE_KEYWORDS.get(role_key, [])
        missing_skills = [kw for kw in role_kws if kw.lower() not in cv_text.lower()][:10]
        
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
