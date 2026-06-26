from typing import Dict, Any, List, Tuple
import json
import re

class EvidenceExtractor:
    """
    Extracts production-grade evidence from candidate descriptions.
    Uses a weighted taxonomy to score the "Shipper" quality of a profile.
    """
    def __init__(self, spec_path: str):
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        self.taxonomy = self.spec.get('production_evidence_keyword_taxonomy', {})
        self.tier_a = self.taxonomy.get('tier_A_highest_value', {})
        self.tier_b = self.taxonomy.get('tier_B_strong_signal', {})
        self.tier_c = self.taxonomy.get('tier_C_weak_positive', {})
        self.negative = self.taxonomy.get('negative_evidence', {})

    def extract_evidence(self, candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Mines career history for evidence and computes a production score.
        Returns (score, list_of_evidence_phrases).
        """
        history = candidate.get('career_history', [])
        if not history:
            return 0.0, []

        total_score = 0.0
        found_evidence = []

        # Aggregate all descriptions for a global scan
        all_text = " ".join([job.get('description', '').lower() for job in history])

        # Tier A: Highest Value (Metrics, Scale, Latency)
        for kw in self.tier_a.get('keywords', []):
            if kw.lower() in all_text:
                # Use regex to capture patterns like "reduced latency by 20%"
                matches = re.findall(rf"{re.escape(kw.lower())}.*?(\d+%)", all_text)
                if matches:
                    found_evidence.append(f"Metric: {kw} ({matches[0]})")
                else:
                    found_evidence.append(kw)
                total_score += self.tier_a.get('weight_per_match', 0.2)

        # Tier B: Strong Signal (Ownership, End-to-End)
        for kw in self.tier_b.get('keywords', []):
            if kw.lower() in all_text:
                found_evidence.append(kw)
                total_score += self.tier_b.get('weight_per_match', 0.1)

        # Tier C: Weak Positive (General Implementation)
        for kw in self.tier_c.get('keywords', []):
            if kw.lower() in all_text:
                total_score += self.tier_c.get('weight_per_match', 0.03)

        # Negative Evidence (Academic/POC/Tutorials)
        for kw in self.negative.get('keywords', []):
            if kw.lower() in all_text:
                total_score += self.negative.get('weight_per_match', -0.05)

        # Cap scores based on taxonomy limits
        score_a = min(sum(self.tier_a.get('weight_per_match', 0.2) for kw in self.tier_a.get('keywords', []) if kw.lower() in all_text),
                      self.tier_a.get('max_contribution', 1.0))
        score_b = min(sum(self.tier_b.get('weight_per_match', 0.1) for kw in self.tier_b.get('keywords', []) if kw.lower() in all_text),
                      self.tier_b.get('max_contribution', 0.5))
        score_c = min(sum(self.tier_c.get('weight_per_match', 0.03) for kw in self.tier_c.get('keywords', []) if kw.lower() in all_text),
                      self.tier_c.get('max_contribution', 0.15))
        penalty = max(sum(self.negative.get('weight_per_match', -0.05) for kw in self.negative.get('keywords', []) if kw.lower() in all_text),
                      self.negative.get('max_penalty', -0.2))

        final_score = score_a + score_b + score_c + penalty
        return float(max(0.0, final_score)), found_evidence
