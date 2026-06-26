from typing import List, Dict, Any, Tuple, Union
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from loguru import logger

# ── Tech availability floor (year the technology became usable in prod) ──────
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

DESCRIPTION_DUPLICATE_THRESHOLD = 0.85
DESCRIPTION_OVERLAP_THRESHOLD = 0.60
UNEXPLAINED_GAP_MONTHS_MAJOR = 9
UNEXPLAINED_GAP_MONTHS_MINOR = 6

class TrustEngine:
    """
    Advanced trust scoring engine that computes a 0-100 trust score
    based on identity, consistency, behavior, market signals, and activity.
    """
    def __init__(self):
        pass

    def _sim(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def _months_ago(self, date_str: str) -> float:
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return (datetime.now() - d).days / 30.44
        except Exception:
            return 9999

    def _score_identity(self, signals: dict) -> float:
        score = 0.0
        if signals.get("verified_email"):     score += 7
        if signals.get("verified_phone"):     score += 7
        if signals.get("linkedin_connected"): score += 6
        return min(score, 20)

    def _score_consistency(self, career: list, skills: list) -> tuple[float, list[str]]:
        score = 25.0
        flags = []
        desc_pairs = [
            (job.get("company", "?"), job.get("description", ""))
            for job in career
            if job.get("description", "").strip()
        ]
        seen_pairs = set()
        for i in range(len(desc_pairs)):
            for j in range(i + 1, len(desc_pairs)):
                key = (min(i, j), max(i, j))
                if key in seen_pairs: continue
                seen_pairs.add(key)
                ratio = self._sim(desc_pairs[i][1], desc_pairs[j][1])
                if ratio >= DESCRIPTION_DUPLICATE_THRESHOLD:
                    score -= 8
                    flags.append(f"DUPLICATE_DESC: '{desc_pairs[i][0]}' ≈ '{desc_pairs[j][0]}' ({ratio:.2f})")
                elif ratio >= DESCRIPTION_OVERLAP_THRESHOLD:
                    score -= 4
                    flags.append(f"OVERLAP_DESC: '{desc_pairs[i][0]}' ~ '{desc_pairs[j][0]}' ({ratio:.2f})")

        current_year = datetime.now().year
        for skill in skills:
            name_lower = skill.get("name", "").lower()
            duration_months = skill.get("duration_months", 0) or 0
            for tech, avail_year in TECH_AVAILABILITY.items():
                if tech in name_lower:
                    implied_start_year = current_year - (duration_months / 12)
                    if implied_start_year < avail_year - 1:
                        score -= 2
                        flags.append(f"SKILL_INFLATION: '{skill['name']}' claimed for {duration_months}m, but {tech} available ~{avail_year}")
                    break

        sorted_jobs = sorted(
            [j for j in career if j.get("start_date") and j.get("end_date")],
            key=lambda x: x["start_date"],
        )
        for i in range(1, len(sorted_jobs)):
            prev_end = sorted_jobs[i - 1]["end_date"]
            curr_start = sorted_jobs[i]["start_date"]
            try:
                end = datetime.strptime(prev_end[:10], "%Y-%m-%d")
                start = datetime.strptime(curr_start[:10], "%Y-%m-%d")
                gap = (start - end).days / 30.44
                if gap > UNEXPLAINED_GAP_MONTHS_MAJOR:
                    score -= 3
                    flags.append(f"GAP: {gap:.0f}m between '{sorted_jobs[i-1]['company']}' and '{sorted_jobs[i]['company']}'")
                elif gap > UNEXPLAINED_GAP_MONTHS_MINOR:
                    score -= 1
                    flags.append(f"MINOR_GAP: {gap:.0f}m between '{sorted_jobs[i-1]['company']}' and '{sorted_jobs[i]['company']}'")
            except Exception: pass
        return max(0.0, min(score, 25)), flags

    def _score_behavioral(self, signals: dict) -> float:
        score = 0.0
        rr = float(signals.get("recruiter_response_rate") or 0)
        score += round(rr * 8)
        icr = float(signals.get("interview_completion_rate") or 0)
        score += round(icr * 7)
        oar = float(signals.get("offer_acceptance_rate") or 0)
        score += round(oar * 6)
        art = float(signals.get("avg_response_time_hours") or 999)
        if art < 12: score += 4
        elif art < 24: score += 3
        elif art < 48: score += 2
        elif art < 72: score += 1
        return min(score, 25)

    def _score_market(self, signals: dict) -> float:
        score = 0.0
        saves = int(signals.get("saved_by_recruiters_30d") or 0)
        if saves >= 30: score += 5
        elif saves >= 20: score += 4
        elif saves >= 10: score += 3
        elif saves >= 5:  score += 2
        elif saves >= 1:  score += 1
        views = int(signals.get("profile_views_received_30d") or 0)
        if views >= 100: score += 4
        elif views >= 50:  score += 3
        elif views >= 20:  score += 2
        elif views >= 5:   score += 1
        endorsements = int(signals.get("endorsements_received") or 0)
        if endorsements >= 30: score += 3
        elif endorsements >= 15: score += 2
        elif endorsements >= 5:  score += 1
        conns = int(signals.get("connection_count") or 0)
        if conns >= 1000: score += 3
        elif conns >= 500:  score += 2
        elif conns >= 200:  score += 1
        return min(score, 15)

    def _score_activity(self, signals: dict) -> float:
        score = 0.0
        last_active = signals.get("last_active_date")
        if last_active:
            days_ago = self._months_ago(last_active) * 30.44
            if days_ago <= 7:  score += 5
            elif days_ago <= 30:  score += 4
            elif days_ago <= 60:  score += 3
            elif days_ago <= 180: score += 2
            elif days_ago <= 365: score += 1
        gh = float(signals.get("github_activity_score") or 0)
        if gh >= 80: score += 4
        elif gh >= 60: score += 3
        elif gh >= 40: score += 2
        elif gh >= 20: score += 1
        pc = float(signals.get("profile_completeness_score") or 0)
        if pc >= 85: score += 3
        elif pc >= 70: score += 2
        elif pc >= 50: score += 1
        assessments = signals.get("skill_assessment_scores") or {}
        if isinstance(assessments, str):
            try: assessments = json.loads(assessments)
            except: assessments = {}
        n = len(assessments)
        avg = sum(assessments.values()) / n if n > 0 else 0
        if n >= 3 and avg >= 75: score += 3
        elif n >= 2 and avg >= 65: score += 2
        elif n >= 1 and avg >= 60: score += 1
        return min(score, 15)

    def analyze(self, candidate: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """
        Returns (total_trust_score, calculation_breakdown, flags).
        """
        signals = candidate.get("redrob_signals") or {}
        career = candidate.get("career_history") or []
        skills = candidate.get("skills") or []

        id_score = self._score_identity(signals)
        con_score, flags = self._score_consistency(career, skills)
        beh_score = self._score_behavioral(signals)
        mkt_score = self._score_market(signals)
        act_score = self._score_activity(signals)

        total = round(id_score + con_score + beh_score + mkt_score + act_score, 1)
        calculation = f"ID:{id_score}|CON:{con_score}|BEH:{beh_score}|MKT:{mkt_score}|ACT:{act_score}"

        return total, calculation, flags
