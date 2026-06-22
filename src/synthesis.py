from typing import Dict, Any, List

class ReasonSynthesizer:
    """
    Generates human-readable, evidence-based justifications for a candidate's rank.
    Deterministic fallback implementation.
    """
    def synthesize(self, result: Dict[str, Any]) -> str:
        """
        Builds a reasoning string based on the score components and evidence.
        """
        cand = result['candidate_data']
        score = result['final_score']
        evidence = result['evidence']
        dq_reasons = result['dq_reasons']

        if score == 0:
            return f"Rejected: {', '.join(dq_reasons) if dq_reasons else 'Does not meet basic requirements'}."

        # Confidence Tiering
        if score >= 1.2:
            tier = "Strong Fit"
        elif score >= 0.8:
            tier = "Possible Fit"
        else:
            tier = "Insufficient Data"

        # Evidence snippet
        evidence_str = f" Key evidence: {', '.join(evidence[:3])}" if evidence else " Limited production evidence."

        # Experience snippet
        exp = cand.get('profile', {}).get('years_of_experience', 0)
        exp_str = f"{exp}y exp"

        # Final summary
        reasoning = f"[{tier}] {exp_str}. {evidence_str}"

        if dq_reasons:
            reasoning += f" Note: {', '.join(dq_reasons)}."

        return reasoning
