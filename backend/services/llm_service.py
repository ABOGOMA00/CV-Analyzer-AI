"""
CV rewrite service implemented fully locally.
No Hugging Face model/API dependency.
"""

import re
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
    "improved", "increased", "reduced", "delivered", "implemented",
    "developed", "designed", "built", "optimized", "automated", "led",
}


def _is_garbled_text(text: str) -> bool:
    # Reject lines that are mostly symbols or contain replacement/encoding noise.
    if not text:
        return True
    bad_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ",.-:/+()&%"))
    ratio = bad_chars / max(len(text), 1)
    if ratio > 0.18:
        return True
    if text.count("&") >= 3:
        return True
    if re.search(r"(?:[A-Za-z]&){3,}[A-Za-z]?", text):
        return True
    if re.search(r"[^\x00-\x7F]", text):
        return True
    return False


def _clean_line(text: str) -> str:
    text = text.replace("\t", " ").replace("•", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.:;])", r"\1", text)
    return text


def _extract_keywords(job_description: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", job_description.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
        "our", "are", "will", "have", "has", "had", "not", "but", "all", "any",
        "job", "role", "team", "work", "years", "year", "experience", "skills",
        "using", "use", "required", "preferred", "candidate", "ability",
        "need", "seeking", "must", "plus", "good", "strong",
    }
    freq = Counter(w for w in words if w not in stop)
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    keywords = []
    for w, _ in ranked:
        if any(ch.isdigit() for ch in w):
            continue
        keywords.append(w)
        if len(keywords) >= limit:
            break
    return keywords


def _looks_like_section_title(line: str) -> bool:
    cleaned = line.strip().strip(":").lower()
    return cleaned in COMMON_SECTION_TITLES


def _extract_cv_points(cv_text: str, limit: int = 16) -> list[str]:
    seen = set()
    points = []
    for raw in cv_text.splitlines():
        cleaned = _clean_line(raw).lstrip("-* ").strip()
        if not cleaned:
            continue

        # Split merged lines into candidate sentences for cleaner bullets.
        candidates = [c.strip() for c in re.split(r"[.;]\s+", cleaned) if c.strip()]
        if not candidates:
            candidates = [cleaned]

        for candidate in candidates:
            if _looks_like_section_title(candidate):
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


def _inject_keyword(text: str, keyword: str) -> str:
    lower = text.lower()
    if keyword and keyword.lower() not in lower and len(text.split()) < 18:
        return f"{text} with {keyword}"
    return text


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith("."):
        text = text[:-1]
    return text


def _rewrite_line(line: str, idx: int, keyword: str) -> str:
    cleaned = _normalize_punctuation(line.strip().lstrip("-*• "))
    if not cleaned:
        return ""

    tokens = cleaned.split()
    first = tokens[0].lower() if tokens else ""
    if first in WEAK_VERBS and len(tokens) > 1:
        cleaned = " ".join(tokens[1:])
        tokens = cleaned.split()
        first = tokens[0].lower() if tokens else ""

    if first in STRONG_STARTERS:
        sentence = _inject_keyword(cleaned, keyword)
    elif first in WEAK_VERBS or first not in {v.lower() for v in ACTION_VERBS}:
        cleaned = _inject_keyword(cleaned, keyword)
        verb = ACTION_VERBS[idx % len(ACTION_VERBS)]
        sentence = f"{verb} {cleaned}"
    else:
        sentence = _inject_keyword(cleaned, keyword)

    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return f"- {sentence}."


def _build_summary(points: list[str], keywords: list[str]) -> str:
    role_hint = "software engineer"
    joined = " ".join(keywords).lower()
    if any(k in joined for k in ("asp.net", ".net", "c#", "angular")):
        role_hint = ".NET full-stack developer"
    elif any(k in joined for k in ("react", "frontend", "javascript", "typescript")):
        role_hint = "frontend developer"
    elif any(k in joined for k in ("django", "flask", "fastapi", "backend", "api")):
        role_hint = "backend developer"

    if keywords:
        top = ", ".join(keywords[:5]).replace("/", " / ")
        return (
            f"Professional Summary: {role_hint.capitalize()} with practical experience "
            f"delivering production features in {top}."
        )
    if points:
        first = _normalize_punctuation(points[0]).lower()
        return f"Professional Summary: Developer focused on {first}."
    return "Professional Summary: Results-driven software developer."


def _build_core_skills(keywords: list[str]) -> str:
    if not keywords:
        return "Core Skills: Backend development, API integration, and performance optimization."
    selected = []
    for k in keywords:
        label = k.replace("/", " / ").upper() if k in {"sql", "api", "apis", "cicd", "ci/cd"} else k.replace("/", " / ")
        selected.append(label)
        if len(selected) == 10:
            break
    return "Core Skills: " + ", ".join(selected) + "."


def generate_rewritten_cv(cv_text: str, job_description: str) -> str:
    """
    Rewrites CV text with a local deterministic strategy.
    Adds stronger bullet style and job-description keywords.
    """
    if not cv_text.strip():
        raise ValueError("CV text cannot be empty.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    points = _extract_cv_points(cv_text)
    if not points:
        raise RuntimeError("CV text has no usable content.")

    keywords = _extract_keywords(job_description)
    header = "ATS-Optimized CV Draft"
    summary = _build_summary(points, keywords)
    skill_line = _build_core_skills(keywords)

    rewritten_lines = [header, "", summary, skill_line, "", "Experience Highlights:"]
    for idx, line in enumerate(points[:12]):
        kw = keywords[idx % len(keywords)] if keywords else ""
        new_line = _rewrite_line(line, idx, kw)
        if new_line:
            rewritten_lines.append(new_line)

    if len(rewritten_lines) <= 6:
        rewritten_lines.extend(
            [
                "- Led development of backend and frontend features aligned with business requirements.",
                "- Implemented maintainable APIs and improved application reliability and performance.",
                "- Collaborated with cross-functional teams to deliver tested, production-ready releases.",
            ]
        )

    if keywords:
        rewritten_lines.extend(
            [
                "",
                "ATS Keyword Alignment:",
                "- " + ", ".join(keywords[:12]).replace("/", " / "),
            ]
        )

    rewritten = "\n".join(rewritten_lines).strip()
    if not rewritten:
        raise RuntimeError("CV rewrite failed to produce output.")
    return rewritten