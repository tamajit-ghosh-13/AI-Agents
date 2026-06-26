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

from typing import Dict, Any, List, Tuple
import json
import re
from src.orchestration.types import Verdict, Evidence

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

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Mines career history for evidence and computes a production score.
        Returns a structured Verdict.
        """
        history = candidate.get('career_history', [])
        if not history:
            return Verdict(
                agent="EvidenceAgent",
                signal="none",
                confidence=1.0,
                reasoning="No career history provided to extract evidence from.",
                score=0.0
            )

        found_evidence_phrases = []

        # Aggregate all descriptions for a global scan
        all_text = " ".join([job.get('description', '').lower() for job in history])

        # Tier A: Highest Value (Metrics, Scale, Latency)
        for kw in self.tier_a.get('keywords', []):
            if kw.lower() in all_text:
                matches = re.findall(rf"{re.escape(kw.lower())}.*?(\d+%)", all_text)
                if matches:
                    found_evidence_phrases.append(f"Metric: {kw} ({matches[0]})")
                else:
                    found_evidence_phrases.append(kw)

        # Tier B: Strong Signal (Ownership, End-to-End)
        for kw in self.tier_b.get('keywords', []):
            if kw.lower() in all_text:
                found_evidence_phrases.append(kw)

        # Score calculation (keeping the existing logic)
        score_a = min(sum(self.tier_a.get('weight_per_match', 0.2) for kw in self.tier_a.get('keywords', []) if kw.lower() in all_text),
                      self.tier_a.get('max_contribution', 1.0))
        score_b = min(sum(self.tier_b.get('weight_per_match', 0.1) for kw in self.tier_b.get('keywords', []) if kw.lower() in all_text),
                      self.tier_b.get('max_contribution', 0.5))
        score_c = min(sum(self.tier_c.get('weight_per_match', 0.03) for kw in self.tier_c.get('keywords', []) if kw.lower() in all_text),
                      self.tier_c.get('max_contribution', 0.15))
        penalty = max(sum(self.negative.get('weight_per_match', -0.05) for kw in self.negative.get('keywords', []) if kw.lower() in all_text),
                      self.negative.get('max_penalty', -0.2))

        final_score = float(max(0.0, score_a + score_b + score_c + penalty))
        normalized_score = min(1.0, final_score)

        # Determine Signal
        if normalized_score > 0.8: signal = "strong"
        elif normalized_score > 0.5: signal = "moderate"
        elif normalized_score > 0.2: signal = "weak"
        else: signal = "none"

        # Map phrases to Evidence objects
        evidence_objects = [
            Evidence(text=phrase, source="career_history_global")
            for phrase in found_evidence_phrases
        ]

        risks = []
        if penalty < -0.1:
            risks.append("Profile contains significant negative evidence (academic/tutorial focus).")

        return Verdict(
            agent="EvidenceAgent",
            signal=signal,
            confidence=0.85 if found_evidence_phrases else 0.5,
            evidence=evidence_objects,
            risks=risks,
            reasoning=f"Found {len(found_evidence_phrases)} production-grade evidence markers. Score: {normalized_score:.2f}",
            score=normalized_score
        )
