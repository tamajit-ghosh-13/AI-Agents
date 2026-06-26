from typing import Dict, Any, List, Tuple
import json

class TitleSieve:
    """
    The Title Sieve is the first stage of the pipeline.
    It performs a fast, O(N) string matching pass to categorize candidates
    and eliminate noise before any expensive semantic or trajectory analysis.
    """
    def __init__(self, spec_path: str):
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        sieve_config = self.spec.get('title_sieve', {})
        self.hard_eliminate = set(sieve_config.get('hard_eliminate', {}).get('titles', []))
        self.conditional_pass = set(sieve_config.get('conditional_pass', {}).get('titles', []))
        self.direct_pass = set(sieve_config.get('direct_pass_with_scrutiny', {}).get('titles', []))
        self.soft_red_flag = set(sieve_config.get('soft_red_flag_titles', {}).get('titles', []))

    def evaluate(self, candidate: Dict[str, Any]) -> Tuple[str, float]:
        """
        Evaluates a candidate based on their primary job title.
        Returns (category, score_penalty).
        """
        # Extract current title
        history = candidate.get('career_history', [])
        if not history:
            return "unknown", 0.0

        # Assume history is reverse chronological (most recent first)
        current_title = history[0].get('title', '')

        # Exact match check
        if current_title in self.hard_eliminate:
            return "hard_reject", 0.0

        if current_title in self.direct_pass:
            return "direct_pass", 0.0

        if current_title in self.conditional_pass:
            return "conditional_pass", -0.10

        if current_title in self.soft_red_flag:
            return "soft_red_flag", -0.10

        # Case-insensitive substring match as fallback
        current_title_lower = current_title.lower()
        for title in self.hard_eliminate:
            if title.lower() in current_title_lower:
                return "hard_reject", 0.0

        for title in self.direct_pass:
            if title.lower() in current_title_lower:
                return "direct_pass", 0.0

        return "unknown", 0.0
