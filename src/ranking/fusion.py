from typing import Dict, List, Any, Tuple
from src.orchestration.types import CandidateEvaluation, Verdict
from src.query.role_intent import RoleIntent

class RankFusion:
    """
    Fuses individual agent verdicts into a final 0–1 score using the five
    declared ScoringPolicy weights. All inputs are normalized to [0, 1] before
    blending; trust_score is divided by 100 here (TrustEngine returns 0–100).
    """

    def __init__(self, intent: RoleIntent):
        self.intent = intent
        self.policy = intent.scoring_policy

    def fuse(self, eval_obj: CandidateEvaluation) -> Tuple[float, str]:
        """
        Computes the final score using a weighted additive blend of all five
        ScoringPolicy dimensions, attenuated by risk:

            final_score = (
                w_rel   * relevance_score       # 0.40
              + w_trust * (trust_score / 100)   # 0.25  ← normalized from 0-100
              + w_avail * availability_score     # 0.20
              + w_arch  * arch_score             # 0.10
              + w_loc   * location_score         # 0.05
            ) * (1 - risk_score) * (1 + reasoning_boost)
        """
        p = self.policy

        # ── Five policy weights ──────────────────────────────────────────────
        w_rel   = p.relevance_weight                # 0.40
        w_trust = p.trust_coherence_weight          # 0.25
        w_avail = p.availability_behavioral_weight  # 0.20
        w_arch  = p.career_archetype_weight         # 0.10
        w_loc   = p.location_availability_weight    # 0.05

        # ── Component scores (all in [0, 1]) ─────────────────────────────────
        relevance_score   = self._compute_relevance(eval_obj.verdicts)

        # trust_score from TrustEngine is 0–100; normalize to 0–1
        trust_01          = eval_obj.trust_score / 100.0

        availability_score = eval_obj.availability_score  # already 0–1

        # Career archetype score comes from TrajectoryAgent verdict
        arch_score = (
            eval_obj.verdicts["TrajectoryAgent"].score
            if "TrajectoryAgent" in eval_obj.verdicts
            else 0.5
        )

        # Location score computed in pipeline._compute_location_score()
        loc_score = eval_obj.location_score  # already 0–1

        risk_score = eval_obj.risk_score  # 0–1

        # ── Weighted additive blend ───────────────────────────────────────────
        weighted = (
            w_rel   * relevance_score
            + w_trust * trust_01
            + w_avail * availability_score
            + w_arch  * arch_score
            + w_loc   * loc_score
        )

        # Reasoning boost: up to 5% for verbose, evidence-rich verdicts
        reasoning_words = (
            sum(len(v.reasoning.split()) for v in eval_obj.verdicts.values())
            if eval_obj.verdicts
            else 0
        )
        reasoning_factor = min(reasoning_words * 0.001, 0.05)

        final_score = weighted * (1.0 - risk_score) * (1.0 + reasoning_factor)
        # Clamp to [0, 1] — weighted sum is already ≤ 1, but risk/boost could
        # produce tiny float overruns
        final_score = min(1.0, max(0.0, final_score))

        breakdown = (
            f"Rel:{relevance_score:.3f}*{w_rel} + Trust:{trust_01:.3f}*{w_trust} + "
            f"Avail:{availability_score:.3f}*{w_avail} + Arch:{arch_score:.3f}*{w_arch} + "
            f"Loc:{loc_score:.3f}*{w_loc} = {weighted:.3f} "
            f"* (1-Risk:{risk_score:.3f}) * ReasonBoost:{reasoning_factor:.3f}"
        )

        return float(final_score), breakdown

    def _compute_relevance(self, verdicts: Dict[str, Verdict]) -> float:
        """
        Aggregates four agent verdicts into a 0–1 relevance sub-score.

        Internal weights (sum to 1.0):
          TechnicalDepthAgent  → 0.40  (hard-skill match, production evidence)
          SemanticAgent        → 0.35  (embedding similarity to JD — previously unused)
          EvidenceAgent        → 0.15  (quantified achievements)
          ProductContextAgent  → 0.10  (company tier / product DNA)

        TrajectoryAgent is now a top-level weight (w_arch = 0.10) and is NOT
        included in this sub-score to avoid double-counting.
        """
        if not verdicts:
            return 0.0

        weight_map = {
            "TechnicalDepthAgent": 0.40,
            "SemanticAgent":       0.35,  # wired in — was computed but discarded before
            "EvidenceAgent":       0.15,
            "ProductContextAgent": 0.10,
        }

        total_weighted_score = 0.0
        total_weight = 0.0

        for agent_name, weight in weight_map.items():
            if agent_name in verdicts:
                total_weighted_score += verdicts[agent_name].score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight
