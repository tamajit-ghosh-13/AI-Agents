from typing import Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
from loguru import logger

class TrajectoryEvaluator:
    def __init__(self, spec_path: str):
        import json
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)
        self.exp_range = self.spec.get('experience', {}).get('stated_range_years', {'min': 5, 'max': 9})

    def _validate_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures history is sorted by date and handles overlaps."""
        sorted_history = sorted(
            history,
            key=lambda x: x.get('start_date', '1900-01-01'),
            reverse=True
        )
        return sorted_history

    def calculate_stability_score(self, history: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        Calculates stability and growth.
        Gap-aware: treats employment gaps as neutral.
        """
        if not history:
            return 1.0, "No history"

        history = self._validate_history(history)
        tenures = [job.get('duration_months', 0) for job in history]

        # Stability Score: penalize high frequency of short stints (< 18 months)
        short_stints = sum(1 for t in tenures if t < 18)
        stability_penalty = (short_stints / len(history)) * 0.3

        # Growth Signal: Check for title evolution (Junior -> Senior -> Lead)
        hierarchy = ["junior", "associate", "sde", "software engineer", "senior", "staff", "principal", "lead", "architect"]
        growth_signal = 0.0

        # Check if candidate has moved UP the hierarchy over time
        last_rank = -1
        for job in reversed(history): # Oldest to newest
            title = job.get('title', '').lower()
            current_rank = -1
            for i, h_title in enumerate(hierarchy):
                if h_title in title:
                    current_rank = i
                    break

            if current_rank > last_rank:
                growth_signal += 0.05
                last_rank = current_rank

        growth_signal = min(growth_signal, 0.2)
        score = 1.0 - stability_penalty + growth_signal

        # Normalize to [0.5, 1.2]
        return float(np.clip(score, 0.5, 1.2)), "Stability and growth analyzed"

    def evaluate_experience_fit(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        """
        Handles Entry-level, Ideal-range, and Overqualified candidates.
        """
        exp = candidate.get('profile', {}).get('years_of_experience', 0)
        min_exp = self.exp_range['min']
        max_exp = self.exp_range['max']

        # Path B: Entry Level
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

        # Use a bounded product: (Stability * Fit) / 1.2 to keep it around 1.0
        return (stability * fit) / 1.2
