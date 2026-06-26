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

from typing import Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
from loguru import logger
from src.orchestration.types import Verdict, Evidence

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

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Analyzes career trajectory and experience fit to produce a structured Verdict.
        """
        history = candidate.get('career_history', [])

        # 1. Stability & Growth
        stability_score, stability_reason = self._calculate_stability_score(history)

        # 2. Experience Fit
        fit_score, fit_reason = self._evaluate_experience_fit(candidate)

        # Compute normalized score [0.0, 1.0]
        # Original formula: (stability * fit) / 1.2
        raw_score = (stability_score * fit_score) / 1.2
        normalized_score = float(np.clip(raw_score, 0.0, 1.0))

        # Determine Signal
        if normalized_score > 0.8: signal = "strong"
        elif normalized_score > 0.5: signal = "moderate"
        elif normalized_score > 0.2: signal = "weak"
        else: signal = "none"

        # Evidence & Reasoning
        evidence = []
        if history:
            evidence.append(Evidence(text=f"Trajectory stability: {stability_score:.2f}", source="career_history"))

        risks = []
        if fit_score < 1.0 and "flight risk" in fit_reason:
            risks.append("Overqualified: Potential flight risk or compensation mismatch.")
        if stability_score < 0.7:
            risks.append("High job-hopping frequency (multiple short stints).")

        return Verdict(
            agent="TrajectoryAgent",
            signal=signal,
            confidence=0.8 if history else 0.4,
            evidence=evidence,
            risks=risks,
            reasoning=f"{fit_reason}. {stability_reason}. Final normalized score: {normalized_score:.2f}",
            score=normalized_score
        )

    def _calculate_stability_score(self, history: List[Dict[str, Any]]) -> Tuple[float, str]:
        if not history:
            return 1.0, "No history"

        history = self._validate_history(history)
        tenures = [job.get('duration_months', 0) for job in history]

        short_stints = sum(1 for t in tenures if t < 18)
        stability_penalty = (short_stints / len(history)) * 0.3

        hierarchy = ["junior", "associate", "sde", "software engineer", "senior", "staff", "principal", "lead", "architect"]
        growth_signal = 0.0

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

        return float(np.clip(score, 0.5, 1.2)), "Stability and growth analyzed"

    def _evaluate_experience_fit(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        exp = candidate.get('profile', {}).get('years_of_experience', 0)
        min_exp = self.exp_range['min']
        max_exp = self.exp_range['max']

        if exp < 2:
            return 0.8, "Entry-level path: Potential-based evaluation"

        if min_exp <= exp <= max_exp:
            return 1.2, "Ideal experience range"

        if exp > max_exp + 3:
            return 0.9, "Overqualified: Potential flight risk/comp mismatch"

        return 1.0, "Outside ideal range but acceptable"
