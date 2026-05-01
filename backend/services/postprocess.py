import re
from typing import Mapping


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def adjust_predicted_role(
    cv_text: str,
    predicted_role: str,
    all_scores: Mapping[str, float],
) -> str:
    """
    Fast, deterministic post-processing rules to fix the most common confusions.

    Rules are conservative:
    - Only act when we see strong keywords, OR
    - The target class score is close to the current predicted score.
    """
    text = (cv_text or "").lower()

    def score(role: str) -> float:
        try:
            return float(all_scores.get(role, 0.0))
        except Exception:
            return 0.0

    current_score = score(predicted_role)

    def close_enough(role: str, margin: float = 6.0, min_alt: float = 6.0) -> bool:
        # Conservative "near tie" guard.
        alt = score(role)
        return alt >= min_alt and abs(current_score - alt) <= margin

    def hit_count(patterns: list[str]) -> int:
        return sum(1 for p in patterns if re.search(p, text))

    def has_strong(patterns: list[str]) -> bool:
        return _has_any(text, patterns)

    # ARTS <-> TEACHER
    teacher_kw = [
        r"\bteacher\b", r"\bclassroom\b", r"\bcurriculum\b", r"\blesson\b",
        r"\bstudent\b", r"\btutor\b", r"\bschool\b",
    ]
    arts_kw = [
        r"\bartist\b", r"\btheatre\b|\btheater\b", r"\bperform", r"\bactor\b",
        r"\bmusician\b|\bmusic\b", r"\bfine arts\b", r"\bvisual arts\b",
    ]
    teacher_strong = [r"\bteacher\b", r"\bclassroom\b", r"\bcurriculum\b"]
    arts_strong = [r"\bartist\b", r"\btheatre\b|\btheater\b", r"\bmusician\b"]
    if (
        predicted_role == "ARTS"
        and (has_strong(teacher_strong) or hit_count(teacher_kw) >= 2)
        and close_enough("TEACHER", margin=10.0, min_alt=7.5)
    ):
        return "TEACHER"
    if (
        predicted_role == "TEACHER"
        and (has_strong(arts_strong) or hit_count(arts_kw) >= 2)
        and close_enough("ARTS", margin=10.0, min_alt=7.5)
    ):
        return "ARTS"

    # SALES <-> BUSINESS-DEVELOPMENT
    bd_kw = [
        r"\bbusiness development\b", r"\blead generation\b", r"\bpartnership",
        r"\bprospect", r"\baccount executive\b", r"\bpipeline\b", r"\bhunter\b",
    ]
    sales_kw = [
        r"\bquota\b", r"\bclosing\b", r"\bcold call", r"\bupsell\b|\bcross[- ]sell\b",
        r"\bterritory\b", r"\bsalesforce\b", r"\bcrm\b",
    ]
    bd_strong = [r"\bbusiness development\b", r"\blead generation\b", r"\bpartnership"]
    sales_strong = [r"\bquota\b", r"\bsalesforce\b", r"\bcold call"]
    if (
        predicted_role == "SALES"
        and (has_strong(bd_strong) or hit_count(bd_kw) >= 2)
        and close_enough("BUSINESS-DEVELOPMENT", margin=8.0, min_alt=7.0)
    ):
        return "BUSINESS-DEVELOPMENT"
    if (
        predicted_role == "BUSINESS-DEVELOPMENT"
        and (has_strong(sales_strong) or hit_count(sales_kw) >= 2)
        and close_enough("SALES", margin=8.0, min_alt=7.0)
    ):
        return "SALES"

    # APPAREL (often retail) <-> SALES
    retail_kw = [
        r"\bretail\b", r"\bstore\b", r"\bmerchandis", r"\bcashier\b", r"\bpos\b",
        r"\bfashion\b", r"\bapparel\b", r"\bboutique\b",
    ]
    b2b_kw = [
        r"\bb2b\b", r"\baccount management\b", r"\benterprise\b", r"\bsalesforce\b",
        r"\bquota\b", r"\bsaas\b", r"\blead generation\b",
    ]
    if (
        predicted_role == "APPAREL"
        and (has_strong([r"\bb2b\b", r"\bsaas\b", r"\bquota\b"]) or hit_count(b2b_kw) >= 2)
        and close_enough("SALES", margin=10.0, min_alt=7.0)
    ):
        return "SALES"
    if (
        predicted_role == "SALES"
        and (has_strong([r"\bretail\b", r"\bfashion\b", r"\bapparel\b"]) or hit_count(retail_kw) >= 2)
        and close_enough("APPAREL", margin=10.0, min_alt=7.0)
    ):
        return "APPAREL"

    # FINANCE <-> ACCOUNTANT
    finance_kw = [
        r"\bvaluation\b", r"\bdcf\b", r"\bportfolio\b", r"\binvestment\b",
        r"\bequity\b", r"\bcredit\b", r"\brisk\b", r"\bbloomberg\b",
        r"\bfinancial model", r"\bprivate equity\b|\bventure capital\b",
    ]
    acct_kw = [
        r"\bgaap\b|\bifrs\b", r"\bbookkeep", r"\breconcil", r"\baccounts payable\b|\baccounts receivable\b",
        r"\bap\b|\bar\b", r"\bjournal entry\b", r"\bledger\b", r"\btax\b",
        r"\bquickbooks\b|\bsap\b",
    ]
    finance_strong = [r"\bdcf\b", r"\bvaluation\b", r"\bportfolio\b", r"\binvestment\b"]
    acct_strong = [r"\bgaap\b|\bifrs\b", r"\bbookkeep", r"\breconcil", r"\baccounts payable\b|\baccounts receivable\b"]
    if (
        predicted_role == "FINANCE"
        and (has_strong(acct_strong) or hit_count(acct_kw) >= 2)
        and close_enough("ACCOUNTANT", margin=8.0, min_alt=7.5)
    ):
        return "ACCOUNTANT"
    if (
        predicted_role == "ACCOUNTANT"
        and (has_strong(finance_strong) or hit_count(finance_kw) >= 2)
        and close_enough("FINANCE", margin=8.0, min_alt=7.5)
    ):
        return "FINANCE"

    # DIGITAL-MEDIA <-> INFORMATION-TECHNOLOGY
    media_kw = [
        r"\bseo\b", r"\bgoogle ads\b|\bppc\b", r"\bfacebook ads\b|\bmeta ads\b",
        r"\bsocial media\b", r"\bcontent\b", r"\bcopywriting\b", r"\bbrand\b",
        r"\bmarketing\b", r"\bsemrush\b|\bahrefs\b",
    ]
    it_kw = [
        r"\bdevops\b", r"\bbackend\b|\bfrontend\b", r"\bapi\b", r"\bdatabase\b|\bsql\b",
        r"\bpython\b|\bjava\b|\bc\+\+\b|\bc#\b", r"\bdocker\b|\bkubernetes\b",
    ]
    media_strong = [r"\bseo\b", r"\bgoogle ads\b|\bppc\b", r"\bsocial media\b", r"\bcopywriting\b"]
    it_strong = [r"\bdocker\b", r"\bkubernetes\b", r"\bapi\b", r"\bdatabase\b|\bsql\b", r"\bpython\b"]
    if (
        predicted_role == "INFORMATION-TECHNOLOGY"
        and (has_strong(media_strong) or hit_count(media_kw) >= 2)
        and close_enough("DIGITAL-MEDIA", margin=10.0, min_alt=7.0)
    ):
        return "DIGITAL-MEDIA"
    if (
        predicted_role == "DIGITAL-MEDIA"
        and (has_strong(it_strong) or hit_count(it_kw) >= 2)
        and close_enough("INFORMATION-TECHNOLOGY", margin=10.0, min_alt=7.0)
    ):
        return "INFORMATION-TECHNOLOGY"

    return predicted_role
