import json
import re
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
from typing import Union

# ── Tech availability floor (year the technology became usable in prod) ──────
# Claiming expert-level experience starting before these dates is a flag.
TECH_AVAILABILITY: dict[str, int] = {
    "rag": 2021,
    "pgvector": 2021,
    "qdrant": 2021,
    "weaviate": 2020,
    "opensearch": 2021,
    "llamaindex": 2022,
    "llama index": 2022,
    "langchain": 2022,
    "qlora": 2023,
    "lora": 2021,
    "stable diffusion": 2022,
    "gpt-4": 2023,
    "llms": 2020,
    "large language models": 2020,
    "chatgpt": 2022,
    "mistral": 2023,
    "llama2": 2023,
    "pinecone": 2019,
    "sentence transformers": 2019,
    "faiss": 2019,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
DESCRIPTION_DUPLICATE_THRESHOLD  = 0.85   # similarity ratio → major flag
DESCRIPTION_OVERLAP_THRESHOLD    = 0.60   # similarity ratio → minor flag
UNEXPLAINED_GAP_MONTHS_MAJOR     = 9      # gap in months → penalty
UNEXPLAINED_GAP_MONTHS_MINOR     = 6

def _safe_parse(value) -> Union[dict, list, None]:
    """Parse value that may already be a dict/list or a JSON string."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return None

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def _months_ago(date_str: str) -> float:
    """Return how many months ago a date string (YYYY-MM-DD) was."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (datetime.now() - d).days / 30.44
    except Exception:
        return 9999

def _years_ago(date_str: str) -> float:
    return _months_ago(date_str) / 12

# ─────────────────────────────────────────────────────────────────────────────
#  Dimension scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_identity(signals: dict) -> float:
    """Max 20 pts — all-or-nothing per verification type."""
    score = 0.0
    if signals.get("verified_email"):     score += 7
    if signals.get("verified_phone"):     score += 7
    if signals.get("linkedin_connected"): score += 6
    return min(score, 20)

def _score_consistency(career: list, skills: list) -> tuple[float, list[str]]:
    """
    Max 25 pts.
    Returns (score, list_of_flag_messages).
    Starts at 25 and deducts for detected anomalies.
    """
    score = 25.0
    flags = []
    # ── 2a. Duplicate job descriptions ───────────────────────────────────────
    desc_pairs = [
        (job.get("company", "?"), job.get("description", ""))
        for job in career
        if job.get("description", "").strip()
    ]
    seen_pairs = set()
    for i in range(len(desc_pairs)):
        for j in range(i + 1, len(desc_pairs)):
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            ratio = _sim(desc_pairs[i][1], desc_pairs[j][1])
            if ratio >= DESCRIPTION_DUPLICATE_THRESHOLD:
                score -= 8
                flags.append(
                    f"DUPLICATE_DESC: '{desc_pairs[i][0]}' ≈ '{desc_pairs[j][0]}' "
                    f"(similarity={ratio:.2f})"
                )
            elif ratio >= DESCRIPTION_OVERLAP_THRESHOLD:
                score -= 4
                flags.append(
                    f"OVERLAP_DESC: '{desc_pairs[i][0]}' ~ '{desc_pairs[j][0]}' "
                    f"(similarity={ratio:.2f})"
                )
    # ── 2b. Skill-Experience Consistency ──────────────────────────────────────
    current_year = datetime.now().year
    for skill in skills:
        name_lower = skill.get('name', '').lower()
        duration_months = skill.get('duration_months', 0) or 0
        for tech, avail_year in TECH_AVAILABILITY.items():
            if tech in name_lower:
                implied_start_year = current_year - (duration_months / 12)
                if implied_start_year < avail_year - 1:
                    score -= 2
                    flags.append(
                        f"SKILL_INFLATION: '{skill['name']}' claimed for "
                        f"{duration_months}m (implies ~{implied_start_year:.0f}), "
                        f"but {tech} available ~{avail_year}"
                    )
                break
    # ── 2c. Timeline gaps ────────────────────────────────────────────────────
    sorted_jobs = sorted(
        [j for j in career if j.get('start_date') and j.get('end_date')],
        key=lambda x: x['start_date']
    )
    for i in range(1, len(sorted_jobs)):
        prev_end   = sorted_jobs[i - 1]['end_date']
        curr_start = sorted_jobs[i]['start_date']
        try:
            end   = datetime.strptime(prev_end[:10], "%Y-%m-%d")
            start = datetime.strptime(curr_start[:10], "%Y-%m-%d")
            gap   = (start - end).days / 30.44
            if gap > UNEXPLAINED_GAP_MONTHS_MAJOR:
                score -= 3
                flags.append(
                    f"GAP: {gap:.0f}m between "
                    f"'{sorted_jobs[i-1]['company']}' and '{sorted_jobs[i]['company']}'"
                )
            elif gap > UNEXPLAINED_GAP_MONTHS_MINOR:
                score -= 1
                flags.append(
                    f"MINOR_GAP: {gap:.0f}m between "
                    f"'{sorted_jobs[i-1]['company']}' and '{sorted_jobs[i]['company']}'"
                )
        except Exception:
            pass
    return max(0.0, min(score, 25)), flags

def _score_behavioral(signals: dict) -> float:
    """Max 25 pts — recruiter engagement metrics."""
    score = 0.0
    rr = float(signals.get('recruiter_response_rate') or 0)
    score += round(rr * 8)
    icr = float(signals.get('interview_completion_rate') or 0)
    score += round(icr * 7)
    oar = float(signals.get('offer_acceptance_rate') or 0)
    score += round(oar * 6)
    art = float(signals.get('avg_response_time_hours') or 999)
    if   art < 12: score += 4
    elif art < 24: score += 3
    elif art < 48: score += 2
    elif art < 72: score += 1
    return min(score, 25)

def _score_market(signals: dict) -> float:
    """Max 15 pts — external demand signals."""
    score = 0.0
    saves = int(signals.get('saved_by_recruiters_30d') or 0)
    if   saves >= 30: score += 5
    elif saves >= 20: score += 4
    elif saves >= 10: score += 3
    elif saves >= 5:  score += 2
    elif saves >= 1:  score += 1
    views = int(signals.get('profile_views_received_30d') or 0)
    if   views >= 100: score += 4
    elif views >= 50:  score += 3
    elif views >= 20: score += 2
    elif views >= 5:   score += 1
    endorsements = int(signals.get('endorsements_received') or 0)
    if   endorsements >= 30: score += 3
    elif endorsements >= 15: score += 2
    elif endorsements >= 5:  score += 1
    conns = int(signals.get('connection_count') or 0)
    if   conns >= 1000: score += 3
    elif conns >= 500:  score += 2
    elif conns >= 200:  score += 1
    return min(score, 15)

def _score_activity(signals: dict) -> float:
    """Max 15 pts — platform recency and engagement."""
    score = 0.0
    last_active = signals.get('last_active_date')
    if last_active:
        days_ago = _months_ago(last_active) * 30.44
        if   days_ago <=  7:  score += 5
        elif days_ago <= 30:  score += 4
        elif days_ago <= 60:  score += 3
        elif days_ago <= 180: score += 2
        elif days_ago <= 365: score += 1
    gh = float(signals.get('github_activity_score') or 0)
    if   gh >= 80: score += 4
    elif gh >= 60: score += 3
    elif gh >= 40: score += 2
    elif gh >= 20: score += 1
    pc = float(signals.get('profile_completeness_score') or 0)
    if   pc >= 85: score += 3
    elif pc >= 70: score += 2
    elif pc >= 50: score += 1
    assessments = signals.get('skill_assessment_scores') or {}
    if isinstance(assessments, str):
        assessments = _safe_parse(assessments) or {}
    n   = len(assessments)
    avg = sum(assessments.values()) / n if n > 0 else 0
    if   n >= 3 and avg >= 75: score += 3
    elif n >= 2 and avg >= 65: score += 2
    elif n >= 1 and avg >= 60: score += 1
    return min(score, 15)

# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_trust_score(candidate: Union[dict, str]) -> dict:
    """Compute trust score for a single candidate.

    Returns a dict with keys:
        trust_score (0‑100) and sub‑dimension scores.
    """
    if isinstance(candidate, str):
        candidate = _safe_parse(candidate) or {}
    signals = _safe_parse(candidate.get('redrob_signals')) or {}
    career  = _safe_parse(candidate.get('career_history')) or []
    skills  = _safe_parse(candidate.get('skills')) or []
    id_score   = _score_identity(signals)
    con_score, flags = _score_consistency(career, skills)
    beh_score  = _score_behavioral(signals)
    mkt_score  = _score_market(signals)
    act_score  = _score_activity(signals)
    total = round(id_score + con_score + beh_score + mkt_score + act_score, 1)
    return {
        'trust_score': total,
        'identity_verification': id_score,
        'profile_consistency': con_score,
        'behavioral_engagement': beh_score,
        'market_validation': mkt_score,
        'platform_activity': act_score,
        'consistency_flags': flags,
    }
}

# Convenience for CSV batches – not used directly in the pipeline but provided.
def add_trust_scores_to_csv(input_path: str, output_path: str, json_col: str = 'profile_json', include_subdimensions: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if json_col not in df.columns:
        raise ValueError(f"Column '{json_col}' not found in CSV")
    results = df[json_col].apply(lambda x: compute_trust_score(x))
    results_df = pd.DataFrame(results.tolist())
    df['trust_score'] = results_df['trust_score']
    df['consistency_flags'] = results_df['consistency_flags'].apply(lambda x: ' | '.join(x) if x else '')
    if include_subdimensions:
        for dim in ['identity_verification','profile_consistency','behavioral_engagement','market_validation','platform_activity']:
            df[f'ts_{dim}'] = results_df[dim]
    df.to_csv(output_path, index=False)
    return df
