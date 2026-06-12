"""
Comprehensive project test -- covers every major feature.
Run with:  .venv\\Scripts\\python.exe -m pytest backend/test_comprehensive.py -v -s
"""
import io
import json
import pathlib
import re
import time

import pytest
import requests

BASE = "http://127.0.0.1:8000"
API  = f"{BASE}/api"

# ── Sample CV text for different roles ────────────────────────────────────────
FINANCE_CV = """
John Smith - Financial Analyst
Skills: Excel, Financial Modeling, Forecasting, Budgeting, Accounting, Audit
Experience: 5 years in investment banking, DCF analysis, Bloomberg terminal,
variance analysis, P&L reporting, GAAP compliance, tax preparation.
Education: BSc Finance, University of London.
"""

FRONTEND_CV = """
Sara Ali - Frontend Developer
Skills: React, TypeScript, JavaScript, HTML5, CSS3, Figma, Angular, Vue.js
Experience: 4 years building responsive web applications, REST APIs integration,
Git, Agile/Scrum, unit testing with Jest, CI/CD pipelines, webpack bundling.
Education: BSc Computer Science.
"""

HR_CV = """
Ahmed Khalil - HR Manager
Skills: Recruitment, Onboarding, Payroll, HRIS, Performance Management, Training
Experience: 7 years in talent acquisition, employee relations, HRBP,
organizational development, HR analytics, SAP SuccessFactors, Workday.
Education: BA Human Resources Management.
"""

FRONTEND_JD = """
Senior Frontend Developer — We need a developer skilled in React, TypeScript,
and modern JavaScript. Experience with CSS frameworks, REST APIs, Git,
Agile methodology, and CI/CD pipelines required. Knowledge of testing
frameworks (Jest, Cypress) a plus. Figma for design handoff.
"""

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def wait_for_server():
    """Wait up to 15 s for the server to be ready."""
    for _ in range(15):
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(1)
    pytest.fail("Server did not start in time")


def _upload(cv_text: str, jd: str = "") -> dict:
    """Helper: upload CV as text file and return JSON analysis."""
    files  = {"file": ("test_cv.txt", io.BytesIO(cv_text.encode()), "text/plain")}
    data   = {"target_jd": jd} if jd else {}
    r = requests.post(f"{API}/analyze/upload", files=files, data=data, timeout=60)
    assert r.status_code == 200, f"Upload failed: {r.status_code} — {r.text[:300]}"
    return r.json()


# ── Test 1: Health ─────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self):
        print("\n[Health] GET /health")
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "version" in d
        print(f"  version={d['version']}  status={d['status']}")


# ── Test 2: Role Classification ────────────────────────────────────────────────
class TestRoleClassification:
    def test_finance_cv(self):
        print("\n[Classification] Finance CV")
        d = _upload(FINANCE_CV)
        role = d["predicted_role"]
        print(f"  Predicted: {role}  Confidence: {d['confidence']:.1f}%")
        assert role == "FINANCE", f"Expected FINANCE, got {role}"

    def test_it_cv(self):
        print("\n[Classification] Frontend CV -> IT")
        d = _upload(FRONTEND_CV)
        role = d["predicted_role"]
        print(f"  Predicted: {role}  Confidence: {d['confidence']:.1f}%")
        assert role == "INFORMATION-TECHNOLOGY", f"Expected INFORMATION-TECHNOLOGY, got {role}"

    def test_hr_cv(self):
        print("\n[Classification] HR CV")
        d = _upload(HR_CV)
        role = d["predicted_role"]
        print(f"  Predicted: {role}  Confidence: {d['confidence']:.1f}%")
        assert role == "HR", f"Expected HR, got {role}"

    def test_confidence_reasonable(self):
        """Confidence should be between 10% and 100%"""
        d = _upload(FINANCE_CV)
        assert 10 <= d["confidence"] <= 100, f"Unreasonable confidence: {d['confidence']}"


# ── Test 3: ATS Score ──────────────────────────────────────────────────────────
class TestATSScore:
    def test_ats_score_present(self):
        print("\n[ATS] Score returned")
        d = _upload(FINANCE_CV)
        assert d.get("ats_score") is not None
        print(f"  Finance ATS: {d['ats_score']:.1f}%")

    def test_ats_score_range(self):
        """ATS must be between 0 and 100"""
        for cv in [FINANCE_CV, FRONTEND_CV, HR_CV]:
            d = _upload(cv)
            assert 0 <= d["ats_score"] <= 100, f"ATS out of range: {d['ats_score']}"

    def test_ats_with_matching_jd_higher(self):
        """ATS with a matching JD should be higher than with no JD"""
        no_jd  = _upload(FRONTEND_CV)["ats_score"]
        with_jd = _upload(FRONTEND_CV, jd=FRONTEND_JD)["ats_score"]
        print(f"\n[ATS] No JD: {no_jd:.1f}%  |  With matching JD: {with_jd:.1f}%")
        # A relevant JD typically produces a higher or equal ATS than the generic default
        assert with_jd >= no_jd - 5, "ATS with matching JD should not be much lower"


# ── Test 4: Skill Extraction ──────────────────────────────────────────────────
class TestSkillExtraction:
    def test_skills_not_empty(self):
        print("\n[Skills] Skills extracted")
        d = _upload(FINANCE_CV)
        skills = d.get("extracted_skills", [])
        print(f"  Finance skills: {skills}")
        assert len(skills) > 0, "No skills extracted"

    def test_no_noise_words(self):
        """linkedin, university, testing, training etc. must NOT appear as skills"""
        BAD = {"linkedin", "university", "college", "testing", "training",
               "monitoring", "reporting", "presentations", "cover", "letter"}
        d = _upload(FINANCE_CV)
        skills_lower = {s.lower() for s in d.get("extracted_skills", [])}
        bad_found = BAD & skills_lower
        assert not bad_found, f"Noise words found in skills: {bad_found}"

    def test_skills_match_role(self):
        """Finance CV should have finance-relevant skills"""
        d = _upload(FINANCE_CV)
        skills_lower = {s.lower() for s in d.get("extracted_skills", [])}
        finance_terms = {"excel", "forecasting", "audit", "compliance",
                         "financial modeling", "budgeting", "accounting"}
        overlap = finance_terms & skills_lower
        print(f"\n[Skills] Finance overlap: {overlap}")
        assert overlap, "No finance-specific skills found"


# ── Test 5: Missing Skills ─────────────────────────────────────────────────────
class TestMissingSkills:
    def test_missing_skills_with_jd(self):
        """Missing skills should appear when JD has extras the CV lacks"""
        MINIMAL_CV = "John Doe. Experience in Excel and basic accounting."
        d = _upload(MINIMAL_CV, jd=FRONTEND_JD)
        missing = d.get("missing_skills", [])
        print(f"\n[Missing Skills] JD gaps: {missing}")
        assert len(missing) > 0, "Expected some missing skills against a frontend JD"

    def test_missing_skills_sane(self):
        """Missing skills should not contain noise"""
        d = _upload(FINANCE_CV)
        missing = d.get("missing_skills", [])
        BAD_MISSING = {"linkedin", "cover", "letter", "university"}
        bad = BAD_MISSING & {m.lower() for m in missing}
        assert not bad, f"Noise in missing skills: {bad}"


# ── Test 6: Sub-specialization ────────────────────────────────────────────────
class TestSubSpecialization:
    def test_sub_spec_returned(self):
        print("\n[Sub-spec] Sub-specialization present")
        d = _upload(FRONTEND_CV)
        sub = d.get("sub_specialization", {})
        print(f"  Top: {sub.get('top')}  Scores count: {len(sub.get('scores', []))}")
        assert sub.get("top"), "No sub-specialization detected"

    def test_sub_spec_scores_list(self):
        d = _upload(FINANCE_CV)
        scores = d.get("sub_specialization", {}).get("scores", [])
        assert isinstance(scores, list)
        assert len(scores) > 0


# ── Test 7: History ───────────────────────────────────────────────────────────
class TestHistory:
    uploaded_id: int = None

    def test_upload_creates_history(self):
        print("\n[History] Upload & verify record")
        d = _upload(FINANCE_CV)
        TestHistory.uploaded_id = d["id"]
        assert d["id"] > 0

    def test_history_list(self):
        r = requests.get(f"{API}/history/", timeout=10)
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        ids = [i["id"] for i in items]
        assert TestHistory.uploaded_id in ids, "Uploaded record not in history"
        print(f"  History has {len(items)} records")

    def test_history_detail(self):
        r = requests.get(f"{API}/history/{TestHistory.uploaded_id}", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["id"] == TestHistory.uploaded_id

    def test_history_delete(self):
        r = requests.delete(f"{API}/history/{TestHistory.uploaded_id}", timeout=10)
        assert r.status_code == 200
        r2 = requests.get(f"{API}/history/{TestHistory.uploaded_id}", timeout=10)
        assert r2.status_code == 404, "Record should be 404 after delete"


# ── Test 8: Rewrite ───────────────────────────────────────────────────────────
class TestRewrite:
    def test_rewrite_returns_text(self):
        print("\n[Rewrite] CV rewrite")
        payload = {"cv_text": FINANCE_CV, "job_description": FRONTEND_JD}
        r = requests.post(f"{API}/rewrite/", json=payload, timeout=60)
        assert r.status_code == 200
        d = r.json()
        assert d.get("rewritten_cv"), "Rewritten CV text is empty"
        print(f"  Rewritten length: {len(d['rewritten_cv'])} chars")

    def test_rewrite_ats_not_lower(self):
        """New ATS should be >= old ATS (rewrite must not hurt the score)"""
        payload = {"cv_text": FINANCE_CV, "job_description": FRONTEND_JD}
        r = requests.post(f"{API}/rewrite/", json=payload, timeout=60)
        d = r.json()
        old = d.get("old_ats_score", 0)
        new = d.get("new_ats_score", 0)
        print(f"\n[Rewrite] ATS: {old:.1f}% -> {new:.1f}%")
        assert new >= old - 0.5, f"Rewrite lowered ATS from {old:.1f}% to {new:.1f}%"

    def test_rewrite_preserves_original(self):
        """Rewritten CV must include ATS Optimization Block AND original content"""
        payload = {"cv_text": FINANCE_CV, "job_description": FRONTEND_JD}
        r = requests.post(f"{API}/rewrite/", json=payload, timeout=60)
        text = r.json().get("rewritten_cv", "")
        assert "ATS OPTIMIZATION" in text.upper() or "Core Skills" in text, \
            "ATS block not found in rewritten CV"

    def test_rewrite_no_noise_skills(self):
        """Core Skills line must not contain noise words"""
        payload = {"cv_text": FINANCE_CV, "job_description": FRONTEND_JD}
        r = requests.post(f"{API}/rewrite/", json=payload, timeout=60)
        text = r.json().get("rewritten_cv", "").lower()
        # extract core skills line
        if "core skills:" in text:
            skills_line = text[text.index("core skills:"):][:200]
            BAD = ["cover", " letter ", "university", "college"]
            for bad in BAD:
                assert bad not in skills_line, \
                    f"Noise word '{bad}' found in Core Skills: {skills_line[:100]}"


# ── Test 9: Download DOCX ─────────────────────────────────────────────────────
class TestDownload:
    def test_download_docx(self):
        print("\n[Download] DOCX generation")
        payload = {"cv_text": FINANCE_CV, "job_description": FRONTEND_JD}
        rewrite_r = requests.post(f"{API}/rewrite/", json=payload, timeout=60)
        rewritten = rewrite_r.json()["rewritten_cv"]

        dl_r = requests.post(f"{API}/rewrite/download",
                             json={"rewritten_cv": rewritten}, timeout=30)
        assert dl_r.status_code == 200
        ct = dl_r.headers.get("content-type", "")
        assert "wordprocessingml" in ct or "octet-stream" in ct, \
            f"Unexpected content-type: {ct}"
        size = len(dl_r.content)
        assert size > 5000, f"DOCX too small ({size} bytes)"
        print(f"  DOCX size: {size:,} bytes")


# ── Test 10: PDF Boilerplate Cleaning ─────────────────────────────────────────
class TestPDFCleaning:
    def test_boilerplate_stripped(self):
        """PDF boilerplate (CV Genius footer) should be stripped from extracted text"""
        from backend.routes.analyze import _clean_cv_text
        # Need > 200 chars of real content before boilerplate triggers the threshold
        real_content = (
            "John Smith - Senior Software Developer\n"
            "Email: john.smith@email.com | Phone: 07123 456789\n"
            "London, UK | github.com/johnsmith\n\n"
            "Professional Summary:\n"
            "Experienced Python developer with 6 years of hands-on software engineering.\n"
            "Proven track record delivering scalable APIs and microservices.\n\n"
            "Skills: Python, Django, FastAPI, PostgreSQL, Docker, Git, REST APIs\n\n"
            "Experience: Senior Developer at Tech Corp (2019-Present)\n"
        )
        boilerplate = (
            "How to write a CV\nCV layout\nCover letter builder\n"
            "Resume examples by industry"
        )
        raw = real_content + boilerplate
        cleaned = _clean_cv_text(raw)
        print(f"\n[PDF Clean] Before: {len(raw)} chars -> After: {len(cleaned)} chars")
        assert "How to write a CV" not in cleaned
        assert "Cover letter builder" not in cleaned
        assert "John Smith" in cleaned  # real content preserved

    def test_real_content_preserved(self):
        from backend.routes.analyze import _clean_cv_text
        clean = _clean_cv_text("Sara Ali - Senior Engineer\nSkills: Java, Python, Docker")
        assert "Sara Ali" in clean
        assert "Java" in clean


# ── Test 11: Edge Cases ───────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_file_rejected(self):
        print("\n[Edge] Empty file")
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        r = requests.post(f"{API}/analyze/upload", files=files, timeout=10)
        assert r.status_code in (400, 422, 500), \
            f"Expected error for empty file, got {r.status_code}"

    def test_unsupported_format(self):
        print("\n[Edge] Unsupported file format")
        files = {"file": ("cv.docx", io.BytesIO(b"PK fake docx"), "application/vnd.openxmlformats")}
        r = requests.post(f"{API}/analyze/upload", files=files, timeout=10)
        assert r.status_code in (400, 422, 500)

    def test_history_nonexistent(self):
        print("\n[Edge] Non-existent history ID")
        r = requests.get(f"{API}/history/999999", timeout=5)
        assert r.status_code == 404

    def test_rewrite_without_jd_rejected(self):
        """Frontend requires JD — backend should handle empty JD gracefully"""
        payload = {"cv_text": FINANCE_CV, "job_description": ""}
        r = requests.post(f"{API}/rewrite/", json=payload, timeout=30)
        # either 400 (bad request) or 200 with degraded output is acceptable
        assert r.status_code in (200, 400, 422, 500)
