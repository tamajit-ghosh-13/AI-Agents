from typing import Dict, List, Any, Tuple
from src.orchestration.types import CandidateEvaluation, Verdict
from src.query.role_intent import RoleIntent

class RankFusion:
    """
    Fuses individual agent verdicts into a final numerical score
    using a multiplicative model conditioned on trust and availability.
    """

    def __init__(self, intent: RoleIntent):
        self.intent = intent
        self.policy = intent.scoring_policy

    def fuse(self, eval_obj: CandidateEvaluation) -> Tuple[float, str]:
        """
        Computes the final score using the formula:
        final_score = relevance_score * trust_score * availability_score * (1 - risk_score)
        """
        relevance_score = self._compute_relevance(eval_obj.verdicts)

        # We use the trust_score and availability_score already computed in the pipeline
        trust_score = eval_obj.trust_score
        availability_score = eval_obj.availability_score

        # Risk score is aggregated from high-severity anomalies or hard-reject triggers
        risk_score = eval_obj.risk_score

        # Multiplicative Fusion
        # Base multiplicative fusion
        base_score = relevance_score * trust_score * availability_score * (1.0 - risk_score)

        # Reasoning boost: give higher value to strong reasoning statements.
        # Compute total words across all verdict reasoning texts.
        reasoning_words = sum(len(v.reasoning.split()) for v in eval_obj.verdicts.values()) if eval_obj.verdicts else 0
        reasoning_factor = min(reasoning_words * 0.001, 0.10)  # up to 10% boost
        final_score = base_score * (1.0 + reasoning_factor)

        breakdown = (f"Rel:{relevance_score:.2f} * Trust:{trust_score:.1f} * Avail:{availability_score:.2f} "
                    f"* (1-Risk:{risk_score:.2f}) * ReasonBoost:{reasoning_factor:.3f}")

        return float(final_score), breakdown

    def _compute_relevance(self, verdicts: Dict[str, Verdict]) -> float:
        """
        Aggregates individual agent verdicts into a single relevance score.
        Weights are derived from the RoleIntent's scoring_policy.
        """
        if not verdicts:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        # Mapping agent names to their relative weights in the scoring policy
        weight_map = {
            "TechnicalDepthAgent": self.policy.relevance_weight,     # 0.40
            "TrajectoryAgent": self.policy.career_archetype_weight,  # 0.10
            "ProductContextAgent": 0.1,
            "EvidenceAgent": 0.1,
        }

        for agent_name, verdict in verdicts.items():
            if agent_name in weight_map:
                weight = weight_map[agent_name]
                total_weighted_score += verdict.score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight
