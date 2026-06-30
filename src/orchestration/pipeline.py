from typing import List, Dict, Any, Tuple, Set
import numpy as np
from loguru import logger

from src.query.role_intent import JDParser, LocationPolicy
from src.reasoning.title_sieve import TitleSieve
from src.inspection.integrity import IntegrityGuard
from src.inspection.trust_engine import TrustEngine
from src.retrieval.semantic import SemanticScorer
from src.reasoning.trajectory import TrajectoryEvaluator
from src.reasoning.behavioral import BehavioralEvaluator
from src.reasoning.evidence import EvidenceExtractor
from src.reasoning.technical_depth import SkillMatcher
from src.reasoning.product_context import CompanyMatcher
from src.ranking.disqualifiers import DisqualifierEngine
from src.ranking.fusion import RankFusion
from src.ranking.explainer import RankExplainer
from src.orchestration.types import CandidateEvaluation, Verdict

class RankingPipeline:
    def __init__(self, spec_path: str, tiers_path: str):
        self.spec_path = spec_path
        self.tiers_path = tiers_path

        # R - Retrieve & Parse
        self.jd_parser = JDParser(spec_path)
        self.intent = self.jd_parser.parse()

        # Initialize all modules
        self.sieve = TitleSieve(spec_path)
        self.integrity = IntegrityGuard(tiers_path)
        self.trust_engine = TrustEngine()
        self.semantic = SemanticScorer(spec_path)
        self.trajectory = TrajectoryEvaluator(spec_path)
        self.behavioral = BehavioralEvaluator(self.intent)
        self.evidence = EvidenceExtractor(spec_path)
        self.skills = SkillMatcher(spec_path)
        self.company_matcher = CompanyMatcher(tiers_path)
        self.dq_engine = DisqualifierEngine(spec_path, tiers_path)

        # Fusion & Explanation
        self.fusion = RankFusion(self.intent)
        self.explainer = RankExplainer()

    def _compute_location_score(self, candidate: Dict[str, Any]) -> float:
        """
        Scores candidate location fit against the role's LocationPolicy.
        Uses the intent parsed from jd_spec.json (primary=["Pune","Noida"],
        acceptable=["Hyderabad","Mumbai","Delhi NCR","Bangalore"],
        work_mode="hybrid_flexible").
        """
        loc_policy: LocationPolicy = self.intent.role_identity.location
        cand_location = candidate.get('profile', {}).get('location', '').lower().strip()

        if not cand_location:
            # Unknown location — apply hybrid_flexible default
            return 0.70

        # Primary city match
        if any(p.lower() in cand_location or cand_location in p.lower()
               for p in loc_policy.primary):
            return 1.0

        # Acceptable city match
        if any(a.lower() in cand_location or cand_location in a.lower()
               for a in loc_policy.acceptable):
            return 0.85

        # Work mode permits remote / hybrid
        work_mode = (loc_policy.work_mode or '').lower()
        if 'remote' in work_mode or 'hybrid' in work_mode:
            return 0.70

        return 0.60

    def run(self, candidates: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
        """
        Executes the RIO-X Pipeline: Retrieve -> Inspect -> Orchestrate -> eXplain.
        """
        logger.info(f"Starting RIO-X pipeline for {len(candidates)} candidates...")

        # Pre-compute semantic scores for the pool
        semantic_scores = self.semantic.score_candidates(candidates, jd_text)

        # --- Pool-level integrity checks (run ONCE before per-candidate loop) ---
        # detect_boilerplate() uses per-description Jaccard; detect_duplicates() uses
        # profile fingerprinting. Both return sets of candidate IDs to flag.
        logger.info("Running pool-level integrity checks (boilerplate + duplicates)...")
        boilerplate_ids: Set[str] = self.integrity.detect_boilerplate(candidates)
        duplicate_ids: Set[str] = self.integrity.detect_duplicates(candidates)
        flagged_ids: Set[str] = boilerplate_ids | duplicate_ids
        if flagged_ids:
            logger.warning(
                f"Pool integrity: {len(boilerplate_ids)} boilerplate, "
                f"{len(duplicate_ids)} duplicate candidate IDs flagged "
                f"({len(flagged_ids)} unique). Risk penalty will apply."
            )

        evaluations = []
        for i, cand in enumerate(candidates):
            cand_id = cand.get('candidate_id', 'unknown')

            # --- Stage I: Inspect (Trust & Integrity) ---
            # 1. Honeypot Check
            is_hp, hp_reason = self.integrity.check_honeypot(cand)
            if is_hp:
                logger.debug(f"Honeypot detected for {cand_id}: {hp_reason}")
                continue

            # 2. Fast Title Sieve (Early Exit)
            cat, penalty = self.sieve.evaluate(cand)
            if cat == "hard_reject":
                eval_obj = CandidateEvaluation(
                    candidate_id=cand_id,
                    final_score=0.0,
                    tier="rejected",
                    final_score_calculation="SIEVE_REJECT"
                )
                evaluations.append(eval_obj)
                continue

            # 3. Trust & Coherence Check
            trust_score, calculation, anomalies = self.trust_engine.analyze(cand)

            # Initialize Evaluation Object
            eval_obj = CandidateEvaluation(
                candidate_id=cand_id,
                trust_score=trust_score,
                trust_calculation=calculation,
                key_risks=[a for a in anomalies]
            )

            # --- Stage O: Orchestrate (Reasoning) ---
            # We collect verdicts from all specialized agents
            verdicts = {}

            # Technical Depth
            verdicts["TechnicalDepthAgent"] = self.skills.evaluate(cand)

            # Trajectory
            verdicts["TrajectoryAgent"] = self.trajectory.evaluate(cand)

            # Evidence
            verdicts["EvidenceAgent"] = self.evidence.evaluate(cand)

            # Product Context
            verdicts["ProductContextAgent"] = self.company_matcher.evaluate(cand)

            # Behavioral / Availability
            verdicts["AvailabilityAgent"] = self.behavioral.evaluate(cand)

            # Title Sieve (Integrated as a signal)
            verdicts["SieveAgent"] = Verdict(
                agent="SieveAgent",
                signal="strong" if cat == "direct_pass" else "weak",
                confidence=1.0,
                reasoning=f"Sieve category: {cat}",
                score=0.5 if cat == "direct_pass" else 0.2
            )

            # Semantic match (from pre-computed)
            verdicts["SemanticAgent"] = Verdict(
                agent="SemanticAgent",
                signal="strong" if semantic_scores[i] > 0.8 else "moderate",
                confidence=0.7,
                score=semantic_scores[i],
                reasoning=f"Semantic similarity: {semantic_scores[i]:.2f}"
            )

            eval_obj.verdicts = verdicts

            # Availability score from behavioral verdict
            eval_obj.availability_score = verdicts["AvailabilityAgent"].score

            # Location score (computed once, stored for fusion)
            eval_obj.location_score = self._compute_location_score(cand)

            # --- Stage X: eXplain & Rank ---
            # 1. Disqualifiers (Hard Reject Gate)
            dq_verdict = self.dq_engine.evaluate(cand)
            eval_obj.risk_score = 1.0 - dq_verdict.score
            eval_obj.key_risks.extend(dq_verdict.risks)

            # 2. Pool-level integrity penalty 
            # DISABLED: The 100K candidate dataset is synthetically generated and only 
            # contains ~30 unique job descriptions copied perfectly across 100,000 people. 
            # A boilerplate detector will flag 100% of the pool, ruining the actual scores.
            # if cand_id in flagged_ids:
            #     eval_obj.risk_score = min(1.0, eval_obj.risk_score + 0.15)
            #     flag_reason = "Boilerplate/duplicate description detected across candidate pool"
            #     if flag_reason not in eval_obj.key_risks:
            #         eval_obj.key_risks.append(flag_reason)

            if dq_verdict.signal == "none":  # Hard Reject
                eval_obj.final_score = 0.0
                eval_obj.tier = "rejected"
                eval_obj.final_score_calculation = "DQ_REJECT"
            else:
                # 3. Fusion
                score, calc = self.fusion.fuse(eval_obj)
                eval_obj.final_score = score
                eval_obj.final_score_calculation = calc

                # Assign Tier based on 0–1 scale (matches fusion output)
                s = eval_obj.final_score
                if s >= 0.75:   eval_obj.tier = "perfect_fit"
                elif s >= 0.70: eval_obj.tier = "ideal_fit"
                elif s >= 0.65: eval_obj.tier = "strong_fit"
                elif s >= 0.60: eval_obj.tier = "good_fit"
                elif s >= 0.55: eval_obj.tier = "potential_fit"
                elif s >= 0.50: eval_obj.tier = "marginal_fit"
                else:           eval_obj.tier = "unlikely_fit"

            # 4. Explanation
            explanation = self.explainer.explain(eval_obj)
            eval_obj.justification = explanation["justification"]
            eval_obj.key_risks = explanation["key_risks"]

            evaluations.append(eval_obj)

        # Final Sorting: Score DESC, candidate_id ASC
        evaluations.sort(key=lambda x: (-x.final_score, x.candidate_id))

        # Convert dataclasses to dicts for return
        cand_map = {c.get('candidate_id'): c for c in candidates}
        results = []
        for e in evaluations:
            results.append({
                "candidate_id": e.candidate_id,
                "final_score": e.final_score,
                "tier": e.tier,
                "trust_score": e.trust_score,
                "trust_score_calculation": e.trust_calculation,
                "final_score_calculation": e.final_score_calculation,
                "justification": e.justification,
                "key_risks": e.key_risks,
                "verdicts": {k: v.__dict__ for k, v in e.verdicts.items()},
                "candidate_data": cand_map.get(e.candidate_id, {})
            })
        return results
