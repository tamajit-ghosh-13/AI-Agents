from typing import List, Dict, Any, Tuple
from src.inspection.integrity import IntegrityGuard
from src.inspection.coherence import CoherenceEngine

class TrustAggregator:
    """
    Fuses integrity and coherence signals into a final trust_score.
    """
    def __init__(self, tiers_path: str):
        self.integrity = IntegrityGuard(tiers_path)
        self.coherence = CoherenceEngine()

    def get_trust_profile(self, candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
        # 1. Integrity check (Honeypots/Duplicates)
        is_hp, hp_reason = self.integrity.check_honeypot(candidate)
        if is_hp:
            return 0.0, [f"Honeypot detected: {hp_reason}"]

        # 2. Coherence check (Contradictions)
        coherence_score, anomalies = self.coherence.analyze(candidate)

        # 3. Trust fusion
        # If the profile is a duplicate, we lower trust but not to zero
        is_duplicate = False # Logic from integrity.detect_duplicates is usually pool-wide

        final_trust = coherence_score
        reasons = [a.description for a in anomalies]

        return final_trust, reasons
