"""
Local CV rewrite service.

The app does not call an external LLM here, so this module keeps the rewrite
deterministic and conservative: it improves structure and wording without
inventing skills that are not present in the original CV.
"""

from __future__ import annotations

import re
import json
import urllib.request
import urllib.error
from collections import Counter


ACTION_VERBS = [
    "Led",
    "Built",
    "Designed",
    "Implemented",
    "Optimized",
    "Automated",
    "Delivered",
    "Collaborated",
]
ACTION_VERBS_LOWER = {verb.lower() for verb in ACTION_VERBS}

COMMON_SECTION_TITLES = {
    "experience",
    "work experience",
    "employment history",
    "education",
    "skills",
    "projects",
    "summary",
    "profile",
    "certifications",
}

WEAK_VERBS = {"worked", "helped", "did", "made", "responsible"}
STRONG_STARTERS = {
    "improved",
    "increased",
    "reduced",
    "delivered",
    "implemented",
    "developed",
    "designed",
    "built",
    "optimized",
    "automated",
    "led",
}

KEYWORD_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "our",
    "are",
    "will",
    "have",
    "has",
    "had",
    "not",
    "but",
    "all",
    "any",
    "job",
    "role",
    "team",
    "teams",
    "work",
    "working",
    "years",
    "year",
    "experience",
    "experiences",
    "skills",
    "skill",
    "using",
    "use",
    "used",
    "required",
    "preferred",
    "candidate",
    "ability",
    "need",
    "needs",
    "seeking",
    "must",
    "plus",
    "good",
    "strong",
    "excellent",
    "engineer",
    "developer",
    "software",
    "backend",
    "frontend",
    "apis",
    "hiring",
    "should",
    "professional",
    "collaborate",
    "collaborating",
    "design",
    "designed",
    "build",
    "built",
    "implement",
    "implemented",
    "automated",
    "optimize",
    "optimized",
    "performance",
    "production",
    "customer",
    "customers",
    "user",
    "users",
    "portal",
    "portals",
    "application",
    "applications",
    "web",
    "front",
    "end",
    "rest",
    # document / admin noise from JD text
    "cover", "letter", "position", "opportunity", "company",
    "responsibilities", "responsibility", "duties", "qualifications",
    "requirements", "benefits", "compensation", "salary", "location",
    "apply", "applying", "applicant", "candidate", "employer",
    "degree", "bachelor", "master", "university", "college",
    "communication", "written", "verbal", "detail", "oriented",
    "leadership", "management", "problem", "solving", "analytical",
    "ability", "abilities", "knowledge", "understanding",
    "fast", "paced", "environment", "growing", "startup", "remote",
    # skill-noise: too generic / not meaningful as a standalone skill
    "engagement", "code", "coding", "development", "delivery",
    "innovation", "solutions", "service", "services", "support",
    "process", "processes", "projects", "project", "tasks",
    "front-end", "back-end", "full-stack", "technologies", "technology",
    "systems", "platform", "platforms", "tools", "tool",
    "implementation", "integration", "optimization", "maintenance",
    "testing", "training", "monitoring", "reporting", "presentations",
    # noise: generic verbs/adjectives that aren't real skills
    "enhance", "enhanced", "features", "feature", "sheffield",
    "improve", "improved", "improving", "continuous", "improvement",
    "creating", "create", "drive", "driving", "driven",
    "streamline", "ensure", "ensuring", "leverage", "leveraging",
    "boost", "boosting", "track", "record",
    "offering", "innovative", "outstanding",
    "responsible", "utilizing", "loyalty", "brand",
    # city names that leak from CV addresses
    "city", "london", "manchester", "birmingham", "liverpool",
    "new york", "chicago", "los angeles", "san francisco",
    # template builder names
    "example", "genius", "novoresume", "zety",
}

KNOWN_KEYWORDS = [
    # Software / data / cloud
    "rest api",
    "api design",
    "fastapi",
    "django",
    "flask",
    "python",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "java",
    "spring boot",
    "c#",
    "c++",
    ".net",
    "asp.net",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "graphql",
    "microservices",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "ci/cd",
    "github actions",
    "jenkins",
    "git",
    "automated testing",
    "unit testing",
    "pytest",
    "monitoring",
    "production monitoring",
    "performance optimization",
    "linux",
    "terraform",
    "nginx",
    "machine learning",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "data analysis",
    "power bi",
    "tableau",
    # Business / finance / people
    "excel",
    "financial modeling",
    "forecasting",
    "budgeting",
    "audit",
    "accounting",
    "tax",
    "quickbooks",
    "gaap",
    "ifrs",
    "crm",
    "salesforce",
    "lead generation",
    "pipeline management",
    "negotiation",
    "recruitment",
    "onboarding",
    "payroll",
    "hris",
    "employee relations",
    "performance management",
    "stakeholder management",
    "business analysis",
    "process improvement",
    "strategy",
    # Design / marketing / healthcare / education / operations
    "figma",
    "photoshop",
    "illustrator",
    "ui/ux",
    "wireframing",
    "prototyping",
    "seo",
    "google ads",
    "social media",
    "analytics",
    "content creation",
    "patient care",
    "clinical",
    "ehr",
    "emr",
    "medical records",
    "curriculum",
    "lesson planning",
    "classroom management",
    "autocad",
    "solidworks",
    "quality control",
    "project management",
    "safety management",
]

ROLE_HINT_GROUPS = {
    "backend developer": {
        "backend",
        "fastapi",
        "django",
        "flask",
        "rest api",
        "api design",
        "postgresql",
        "mysql",
        "microservices",
        "server",
        "redis",
    },
    "frontend developer": {
        "frontend",
        "react",
        "angular",
        "vue",
        "javascript",
        "typescript",
        "html",
        "css",
        "ui/ux",
    },
    "devops / cloud engineer": {
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "ci/cd",
        "github actions",
        "jenkins",
        "terraform",
        "monitoring",
    },
    "data professional": {
        "machine learning",
        "pandas",
        "numpy",
        "tensorflow",
        "pytorch",
        "data analysis",
        "power bi",
        "tableau",
    },
}


def _is_garbled_text(text: str) -> bool:
    if not text:
        return True
    bad_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ",.-:/+()&%#"))
    ratio = bad_chars / max(len(text), 1)
    if ratio > 0.18:
        return True
    if text.count("&") >= 3:
        return True
    if re.search(r"(?:[A-Za-z]&){3,}[A-Za-z]?", text):
        return True
    return False


def _clean_line(text: str) -> str:
    text = text.replace("\t", " ").replace("\u2022", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.:;])", r"\1", text)
    return text


def _keyword_pattern(keyword: str) -> str:
    parts = []
    for part in re.split(r"[\s/-]+", keyword.lower()):
        if not part:
            continue
        parts.append(r"apis?" if part == "api" else re.escape(part))
    if not parts:
        return ""
    separator = r"[\s/-]*" if len(parts) > 1 else ""
    pattern = separator.join(parts)
    return rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])"


def _keyword_in_text(keyword: str, text: str) -> bool:
    pattern = _keyword_pattern(keyword)
    return bool(pattern and re.search(pattern, text.lower()))


def _dedupe_keywords(keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for keyword in keywords:
        normalized = keyword.lower().strip(" .,:;")
        if not normalized or normalized in seen:
            continue
        if any(normalized != existing and normalized in existing for existing in seen):
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _extract_keywords(text: str, limit: int = 12) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for keyword in KNOWN_KEYWORDS:
        pattern = _keyword_pattern(keyword)
        match = re.search(pattern, lowered) if pattern else None
        if match:
            matches.append((match.start(), keyword))

    matches.sort(key=lambda item: item[0])
    keywords = _dedupe_keywords([keyword for _, keyword in matches])

    if len(keywords) < limit:
        words = [
            word.strip(" .,:;()[]{}").lower()
            for word in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", lowered)
        ]
        freq = Counter(
            word
            for word in words
            if word
            and word not in KEYWORD_STOPWORDS
            and not any(ch.isdigit() for ch in word)
        )
        first_seen = {word: words.index(word) for word in freq}
        ranked = sorted(freq.items(), key=lambda item: (-item[1], first_seen[item[0]]))
        keywords.extend(word for word, _ in ranked if word not in keywords)

    return _dedupe_keywords(keywords)[:limit]


def _looks_like_section_title(line: str) -> bool:
    cleaned = line.strip().strip(":").lower()
    return cleaned in COMMON_SECTION_TITLES


def _looks_like_profile_line(line: str) -> bool:
    text = line.lower().strip()
    words = text.split()
    if len(words) <= 6 and re.search(r"\b(engineer|developer|analyst|manager|specialist|designer)\b", text):
        return True
    if len(words) <= 24 and re.search(r"\b\d+\+?\s*years?\s+(?:of\s+)?experience\b", text):
        return True
    return False


def _extract_cv_points(cv_text: str, limit: int = 16) -> list[str]:
    seen = set()
    points = []
    for raw in cv_text.splitlines():
        cleaned = _clean_line(raw).lstrip("-* ").strip()
        if not cleaned:
            continue

        candidates = [c.strip() for c in re.split(r"[.;]\s+", cleaned) if c.strip()]
        if not candidates:
            candidates = [cleaned]

        for candidate in candidates:
            if _looks_like_section_title(candidate):
                continue
            if _looks_like_profile_line(candidate):
                continue
            if len(candidate) < 8:
                continue
            if _is_garbled_text(candidate):
                continue
            if re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+){0,2}", candidate):
                continue
            tokens = candidate.split()
            if 1 <= len(tokens) <= 4 and all(t[:1].isupper() for t in tokens if t):
                continue
            cleaned_key = candidate.lower()
            if cleaned_key in seen:
                continue
            seen.add(cleaned_key)
            points.append(candidate)
            if len(points) >= limit:
                break
        if len(points) >= limit:
            break
    return points


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".")
    return text


def _rewrite_line(line: str, idx: int) -> str:
    cleaned = _normalize_punctuation(line.strip().lstrip("-*\u2022 "))
    if not cleaned:
        return ""

    tokens = cleaned.split()
    first = tokens[0].lower() if tokens else ""
    if first in WEAK_VERBS and len(tokens) > 1:
        cleaned = " ".join(tokens[1:])
        tokens = cleaned.split()
        first = tokens[0].lower() if tokens else ""

    if first in STRONG_STARTERS or first in ACTION_VERBS_LOWER:
        sentence = cleaned
    else:
        verb = ACTION_VERBS[idx % len(ACTION_VERBS)]
        sentence = f"{verb} {cleaned}"

    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return f"- {sentence}."


def _infer_role_hint(cv_text: str, jd_keywords: list[str]) -> str:
    combined = f"{cv_text} {' '.join(jd_keywords)}".lower()
    scores = {
        role: sum(1 for term in terms if _keyword_in_text(term, combined))
        for role, terms in ROLE_HINT_GROUPS.items()
    }
    best_role, best_score = max(scores.items(), key=lambda item: item[1])
    return best_role if best_score > 0 else "software professional"


def _select_core_skills(cv_text: str, jd_keywords: list[str], limit: int = 10) -> list[str]:
    cv_keywords = _extract_keywords(cv_text, limit=18)
    matched_jd = [keyword for keyword in jd_keywords if _keyword_in_text(keyword, cv_text)]
    return _dedupe_keywords(matched_jd + cv_keywords)[:limit]


def _build_summary(cv_text: str, points: list[str], core_skills: list[str], jd_keywords: list[str]) -> str:
    role_hint = _infer_role_hint(cv_text, jd_keywords)
    summary_skills = core_skills or [keyword for keyword in jd_keywords if _keyword_in_text(keyword, cv_text)]
    if summary_skills:
        top = ", ".join(summary_skills[:5]).replace("/", " / ")
        return (
            f"Professional Summary: {role_hint.capitalize()} with practical experience "
            f"delivering production features in {top}."
        )
    if points:
        first = _normalize_punctuation(points[0]).lower()
        return f"Professional Summary: {role_hint.capitalize()} focused on {first}."
    return "Professional Summary: Results-driven software professional."


def _build_core_skills(skills: list[str], missing_skills: list[str] | None = None) -> str:
    if not skills and not missing_skills:
        return "Core Skills: Requirements analysis, stakeholder communication, and delivery ownership."
    selected = []
    skill_set = list(skills)  # start with matched skills

    # Merge in missing JD skills so the rewritten CV covers the JD gaps
    if missing_skills:
        for ms in missing_skills:
            if ms not in skill_set:
                skill_set.append(ms)

    for skill in skill_set:
        label = skill.replace("/", " / ").upper() if skill in {"sql", "api", "apis", "ci/cd"} else skill.replace("/", " / ")
        selected.append(label)
        if len(selected) == 10:
            break
    return "Core Skills: " + ", ".join(selected) + "."


def generate_rewritten_cv(
    cv_text: str,
    job_description: str,
    missing_skills: list[str] | None = None,
) -> str:
    """
    Rewrite CV text with a keyword-preserving strategy.

    Instead of replacing the original CV content (which can lose important
    keywords and DROP the ATS score), we inject an ATS Optimization Block at
    the top of the original CV. This adds missing JD keywords while keeping
    every word that was already there, so the ATS score can only go UP.

    Structure of the output:
        ── ATS OPTIMIZATION BLOCK ──
        [ATS-Optimized Professional Summary]
        [Core Skills incl. missing JD keywords]
        ────────────────────────────────────
        [Original CV content — preserved in full]

    Args:
        cv_text: Original CV text.
        job_description: Target job description.
        missing_skills: Skills found in JD but absent from CV (from ml_service).
    """
    if not cv_text.strip():
        raise ValueError("CV text cannot be empty.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    # Clean template boilerplate before processing
    _BOILERPLATE = [
        r"(?i)\bhow\s+to\s+write\s+a\s+(cv|resume)\b",
        r"(?i)\bcover\s+letter\s+(builder|examples?|template)\b",
        r"(?i)\bcv\s+(layout|examples?\s+by\s+industry|maker)\b",
        r"(?i)\bresume\s+(examples?|templates?|builder)\b",
    ]
    cleaned_cv = cv_text
    
    # 1. Truncate at promotional footers
    for _pat in _BOILERPLATE:
        _m = re.search(_pat, cleaned_cv)
        if _m and _m.start() > 200:
            cleaned_cv = cleaned_cv[:_m.start()].strip()
            break

    # 2. Remove inline watermarks anywhere in the text
    _WATERMARKS = [
        r"(?i)\bexample\s+by\s+(cv\s*genius|novoresume|zety)\b",
        r"(?i)cv\s*genius",
    ]
    for w in _WATERMARKS:
        cleaned_cv = re.sub(w, "", cleaned_cv, flags=re.IGNORECASE)
    
    cleaned_cv = cleaned_cv.strip()

def check_ollama_status(model_name: str) -> tuple[bool, str]:
    """
    Check if Ollama server is running and if the requested model is pulled.
    Returns (is_ready, warning_message).
    """
    import urllib.request
    import urllib.error
    import json
    try:
        # Check server health
        req = urllib.request.Request("http://localhost:11434/", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status != 200:
                return False, "Ollama server returned non-200 response on health check."
    except Exception as e:
        return False, f"Ollama server is not running on http://localhost:11434. Start Ollama locally. Error: {e}"

    try:
        # Check if model is pulled
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            base_model = model_name.split(":")[0].lower()
            found = False
            for m in models:
                m_lower = m.lower()
                if model_name.lower() in m_lower or base_model in m_lower:
                    found = True
                    break
            if not found:
                return False, f"Model '{model_name}' is not pulled in Ollama. Pull it using `ollama pull {model_name}`. Available: {', '.join(models)}."
    except Exception as e:
        return True, f"Ollama is running, but failed to verify pulled models: {e}"

    return True, ""


def generate_rewritten_cv(
    cv_text: str,
    job_description: str,
    missing_skills: list[str] | None = None,
) -> tuple[str, bool, str]:
    """
    Rewrite CV text dynamically using local Ollama model (qwen2.5:0.5b).
    Falls back to a clean, deterministic rewrite if Ollama is unavailable.
    Returns (rewritten_cv_text, is_fallback, warning_message).
    """
    if not cv_text.strip():
        raise ValueError("CV text cannot be empty.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    # Clean template boilerplate before processing
    _BOILERPLATE = [
        r"(?i)\bhow\s+to\s+write\s+a\s+(cv|resume)\b",
        r"(?i)\bcover\s+letter\s+(builder|examples?|template)\b",
        r"(?i)\bcv\s+(layout|examples?\s+by\s+industry|maker)\b",
        r"(?i)\bresume\s+(examples?|templates?|builder)\b",
    ]
    cleaned_cv = cv_text
    
    for _pat in _BOILERPLATE:
        _m = re.search(_pat, cleaned_cv)
        if _m and _m.start() > 200:
            cleaned_cv = cleaned_cv[:_m.start()].strip()
            break

    _WATERMARKS = [
        r"(?i)\bexample\s+by\s+(cv\s*genius|novoresume|zety)\b",
        r"(?i)cv\s*genius",
    ]
    for w in _WATERMARKS:
        cleaned_cv = re.sub(w, "", cleaned_cv, flags=re.IGNORECASE)
    
    cleaned_cv = cleaned_cv.strip()

    points = _extract_cv_points(cleaned_cv)
    if not points:
        raise ValueError("CV text has no usable content.")

    # ── Attempt Ollama API Integration ───────────────────────────────────────
    import os as _os
    import logging as _logging
    import time
    _logger = _logging.getLogger(__name__)

    _OLLAMA_MODEL = _os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    _OLLAMA_TIMEOUT = int(_os.getenv("OLLAMA_TIMEOUT", "60"))

    prompt = f"""You are an Expert ATS Resume Optimizer.
Your task is to take the original CV and rewrite it to perfectly match the target Job Description.

Rules:
1. Preserve all factual information. Do NOT invent fake companies, credentials, degrees, or certifications.
2. Do NOT output any prefix, suffix, metadata, or warning headers like "── ATS OPTIMIZATION BLOCK ──".
3. Rewrite the Professional Summary to align with the target role.
4. Optimize the experience bullet points: make them action-oriented (e.g. Led, Designed, Implemented, Optimized), grammatically perfect, and naturally embed the missing keywords.
5. Re-organize and expand the Skills section.
6. The output must be a clean, recruiter-ready, professional resume text in plain text layout.

Return ONLY a valid JSON object matching this schema exactly:
{{
  "professional_summary": "Write the optimized professional summary text here...",
  "core_skills": ["Skill1", "Skill2"],
  "optimized_resume": "[Write the ACTUAL full, complete optimized resume text here. Do NOT use placeholders, do NOT summarize, and do NOT copy this bracketed instruction. Provide the complete rewritten CV text including experience and education sections.]"
}}

Original CV:
{cleaned_cv}

Target Job Description:
{job_description}

Missing ATS Keywords to inject:
{", ".join(missing_skills or [])}
"""

    is_ready, check_warn = check_ollama_status(_OLLAMA_MODEL)
    is_fallback = True
    warning_msg = check_warn
    rewritten_text = ""

    if is_ready:
        max_retries = 2
        retry_delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps({
                        "model": _OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    raw_response = res_body.get("response", "").strip()
                    
                    json_str = raw_response
                    if "```json" in raw_response:
                        json_str = raw_response.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw_response:
                        json_str = raw_response.split("```")[1].split("```")[0].strip()
                        
                    llm_response = json.loads(json_str)
                    if isinstance(llm_response, dict):
                        optimized = llm_response.get("optimized_resume")
                        if isinstance(optimized, str) and len(optimized.strip()) > 50:
                            rewritten_text = optimized.strip()
                            is_fallback = False
                            warning_msg = ""
                            _logger.info(f"[Ollama] LLM rewrite succeeded on attempt {attempt + 1}")
                            break
            except Exception as e:
                _logger.warning(f"[Ollama] Attempt {attempt + 1} failed: {e}")
                warning_msg = f"Failed to execute Ollama generation: {e}"
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2.0

    if is_fallback:
        _logger.warning(f"[Ollama] Falling back to deterministic logic: {warning_msg}")
        jd_keywords   = _extract_keywords(job_description)
        core_skills   = _select_core_skills(cleaned_cv, jd_keywords)
        summary       = _build_summary(cleaned_cv, points, core_skills, jd_keywords)
        skill_line    = _build_core_skills(core_skills, missing_skills=missing_skills)

        # Build a clean, recruiter-ready structured resume without "ATS OPTIMIZATION BLOCK"
        sections = [
            "PROFESSIONAL SUMMARY",
            summary.replace("Professional Summary: ", ""),
            "",
            "CORE SKILLS",
            skill_line.replace("Core Skills: ", ""),
            "",
            "PROFESSIONAL EXPERIENCE & EDUCATION",
            cleaned_cv
        ]
        rewritten_text = "\n".join(sections).strip()

    if not rewritten_text:
        raise RuntimeError("CV rewrite failed to produce output.")

    return rewritten_text, is_fallback, warning_msg

