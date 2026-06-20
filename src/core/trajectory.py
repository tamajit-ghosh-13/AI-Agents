from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np
from loguru import logger

class TrajectoryEvaluator:
    def __init__(self, spec_path: str):
        import json
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)
        self.exp_range = self.spec.get('experience_range', {'min': 5, 'max': 9})

    def calculate_stability_score(self, history: list) -> Tuple[float, str]:
        # \"\"\"
        # Calculates stability and growth.
        # Gap-aware: treats employment gaps as neutral.
        # \"\"\"
        if not history:
            return 1.0, "No history"

        tenures = [job.get('duration_months', 0) for job in history]
        avg_tenure = np.mean(tenures) if tenures else 0

        # Stability Score: penalize high frequency of short stints (< 18 months)
        short_stints = sum(1 for t in tenures if t < 18)
        stability_penalty = (short_stints / len(history)) * 0.3

        # Growth Signal: Check if titles evolve
        titles = [job.get('title', '').lower() for job in history]
        growth_signal = 0.0
        senior_keywords = {'senior', 'lead', 'staff', 'principal', 'head', 'architect'}
        if any(any(kw in t for kw in senior_keywords) for t in titles):
            growth_signal = 0.2

        score = 1.0 - stability_penalty + growth_signal
        return float(np.clip(score, 0.5, 1.5)), "Stability and growth analyzed"

    def evaluate_experience_fit(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        # \"\"\"
        # Handles Entry-level, Ideal-range, and Overqualified candidates.
        # \"\"\"
        exp = candidate.get('profile', {}).get('years_of_experience', 0)
        min_exp = self.exp_range['min']
        max_exp = self.exp_range['max']

        # Path B: Entry Level (Potential-based)
        if exp < 2:
            return 0.8, "Entry-level path: Potential-based evaluation"

        # Ideal Range
        if min_exp <= exp <= max_exp:
            return 1.2, "Ideal experience range"

        # Overqualified (Flight Risk)
        if exp > max_exp + 3:
            return 0.9, "Overqualified: Potential flight risk/comp mismatch"

        return 1.0, "Outside ideal range but acceptable"

    def get_score(self, candidate: Dict[str, Any]) -> float:
        history = candidate.get('career_history', [])
        stability, _ = self.calculate_stability_score(history)
        fit, _ = self.evaluate_experience_fit(candidate)
        return stability * fit
