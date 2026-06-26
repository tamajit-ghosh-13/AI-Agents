from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class Anomaly:
    type: str
    severity: str  # "low", "medium", "high"
    description: str

class CoherenceEngine:
    """
    Analyzes candidate profiles for contradictions, timeline anomalies,
    and consistency issues to determine the trustworthiness of the profile.
    """
    def __init__(self):
        pass

    def analyze(self, candidate: Dict[str, Any]) -> Tuple[float, List[Anomaly]]:
        anomalies = []

        # 1. Title-Summary Alignment
        title = candidate.get('headline', '')
        summary = candidate.get('summary', '').lower()
        if title and summary:
            # Simple example: if title is "Mechanical Engineer" but summary says "AI Expert"
            # In a real system, this would use a small LLM check for semantic contradiction.
            if "mechanical" in title.lower() and "ai" in summary and "machine learning" in summary:
                anomalies.append(Anomaly("Title-Summary Mismatch", "medium", "Headline suggests non-AI domain while summary claims AI expertise."))

        # 2. Timeline Anomaly Detection
        history = candidate.get('career_history', [])
        if history:
            # Sort history by start date (assuming ISO format or similar)
            try:
                # Simplified timeline check
                sorted_history = sorted(history, key=lambda x: x.get('start_date', '0000'))
                for i in range(len(sorted_history) - 1):
                    start_next = sorted_history[i+1].get('start_date', '9999')
                    end_curr = sorted_history[i].get('end_date', '9999')
                    if start_next < end_curr:
                        # Potential overlapping jobs (not always an anomaly, but worth noting)
                        pass
            except Exception as e:
                logger.debug(f"Timeline analysis failed for {candidate.get('candidate_id')}: {e}")

        # 3. Skill-Experience Consistency
        skills = candidate.get('skills', [])
        # Check if "Senior" claims are backed by years of experience
        total_years = self._calculate_total_years(history)
        for skill_obj in skills:
            skill = skill_obj.get('name', '')
            level = skill_obj.get('proficiency', '')
            if (level == "expert" or level == "advanced") and total_years < 2:
                anomalies.append(Anomaly("Skill-Experience Gap", "high", f"Claimed expert/advanced level in {skill} but total experience is < 2 years."))

        # 4. Seniority Trajectory Plausibility
        # e.g., Junior -> Director in 1 year
        if history:
            first_role = history[-1].get('title', '').lower()
            last_role = history[0].get('title', '').lower()
            if "junior" in first_role and "director" in last_role and total_years < 3:
                anomalies.append(Anomaly("Trajectory Anomaly", "high", "Implausible jump from Junior to Director in under 3 years."))

        # Trust Score Calculation
        trust_score = 1.0
        for a in anomalies:
            if a.severity == "high": trust_score -= 0.3
            elif a.severity == "medium": trust_score -= 0.1
            elif a.severity == "low": trust_score -= 0.05

        return max(0.0, trust_score), anomalies

    def _calculate_total_years(self, history: List[Dict[str, Any]]) -> float:
        # Simplified years calculation
        return len(history) * 2.0 # Placeholder for real date subtraction logic
