from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger

from src.query.role_intent import JDParser
from src.reasoning.title_sieve import TitleSieve
from src.inspection.integrity import IntegrityGuard
from src.inspection.coherence import CoherenceEngine
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
        self.coherence = CoherenceEngine()
        self.semantic = SemanticScorer(spec_path)
        self.trajectory = TrajectoryEvaluator(spec_path)
        self.behavioral = BehavioralEvaluator()
        self.evidence = EvidenceExtractor(spec_path)
        self.skills = SkillMatcher(spec_path)
        self.company_matcher = CompanyMatcher(tiers_path)
        self.dq_engine = DisqualifierEngine(spec_path, tiers_path)

        # Fusion & Explanation
        self.fusion = RankFusion(self.intent)
        self.explainer = RankExplainer()

    def run(self, candidates: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
        """
        Executes the RIO-X Pipeline: Retrieve -> Inspect -> Orchestrate -> eXplain.
        """
        logger.info(f"Starting RIO-X pipeline for {len(candidates)} candidates...")

        # Pre-compute semantic scores for the pool
        semantic_scores = self.semantic.score_candidates(candidates, jd_text)

        evaluations = []
        for i, cand in enumerate(candidates):
            cand_id = cand.get('candidate_id', 'unknown')

            # --- Stage I: Inspect (Trust & Integrity) ---
            # 1. Honeypot Check
            is_hp, hp_reason = self.integrity.check_honeypot(cand)
            if is_hp:
                logger.debug(f"Honeypot detected for {cand_id}: {hp_reason}")
                continue

            # 2. Coherence Check
            trust_score, anomalies = self.coherence.analyze(cand)

            # Initialize Evaluation Object
            eval_obj = CandidateEvaluation(
                candidate_id=cand_id,
                trust_score=trust_score,
                key_risks=[a.description for a in anomalies]
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
            cat, penalty = self.sieve.evaluate(cand)
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

            # --- Stage X: eXplain & Rank ---
            # 1. Disqualifiers (Hard Reject Gate)
            dq_verdict = self.dq_engine.evaluate(cand)
            eval_obj.risk_score = 1.0 - dq_verdict.score
            eval_obj.key_risks.extend(dq_verdict.risks)

            if dq_verdict.signal == "none": # Hard Reject
                eval_obj.final_score = 0.0
                eval_obj.tier = "rejected"
            else:
                # 2. Fusion
                eval_obj.final_score = self.fusion.fuse(eval_obj)

                # Assign Tier based on score
                if eval_obj.final_score > 0.8: eval_obj.tier = "strong_fit"
                elif eval_obj.final_score > 0.5: eval_obj.tier = "possible_fit"
                else: eval_obj.tier = "unlikely_fit"

            # 3. Explanation
            explanation = self.explainer.explain(eval_obj)
            eval_obj.justification = explanation["justification"]
            eval_obj.key_risks = explanation["key_risks"]

            evaluations.append(eval_obj)

        # Final Sorting: Score DESC, candidate_id ASC
        evaluations.sort(key=lambda x: (-x.final_score, x.candidate_id))

        # Convert dataclasses to dicts for return
        # Build result dicts with calculation details
        results = []
        for e in evaluations:
            # Construct calculation description from component scores
            component_scores = []
            for name, verdict in e.verdicts.items():
                # Skip sieve if not a score
                if hasattr(verdict, 'score'):
                    component_scores.append(f"{name}:{verdict.score:.2f}")
            calculation = ", ".join(component_scores)
            results.append({
                "candidate_id": e.candidate_id,
                "final_score": e.final_score,
                "tier": e.tier,
                "trust_score": e.trust_score,
                "calculation": calculation,
                "justification": e.justification,
                "key_risks": e.key_risks,
                "verdicts": {k: v.__dict__ for k, v in e.verdicts.items()},
                "candidate_data": next(c for c in candidates if c.get('candidate_id') == e.candidate_id)
            })
        return results
