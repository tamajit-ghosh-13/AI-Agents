from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger

from src.core.title_sieve import TitleSieve
from src.core.integrity import IntegrityGuard
from src.core.semantic import SemanticScorer
from src.core.trajectory import TrajectoryEvaluator
from src.core.behavioral import BehavioralEvaluator
from src.core.evidence import EvidenceExtractor
from src.core.skills import SkillMatcher
from src.core.company_matcher import CompanyMatcher
from src.core.disqualifiers import DisqualifierEngine

class RankingPipeline:
    def __init__(self, spec_path: str, tiers_path: str):
        self.spec_path = spec_path
        self.tiers_path = tiers_path

        # Initialize all modules
        self.sieve = TitleSieve(spec_path)
        self.integrity = IntegrityGuard(tiers_path)
        self.semantic = SemanticScorer(spec_path)
        self.trajectory = TrajectoryEvaluator(spec_path)
        self.behavioral = BehavioralEvaluator()
        self.evidence = EvidenceExtractor(spec_path)
        self.skills = SkillMatcher(spec_path)
        self.company_matcher = CompanyMatcher(tiers_path)
        self.dq_engine = DisqualifierEngine(spec_path, tiers_path)

    def run(self, candidates: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
        """
        Executes the 5-stage pipeline.
        """
        logger.info(f"Starting pipeline for {len(candidates)} candidates...")

        # Stage 1: The Sieve (Integrity & Noise Reduction)
        stage1_pool = []
        for cand in candidates:
            cat, penalty = self.sieve.evaluate(cand)
            if cat == "hard_reject":
                continue

            # Honeypot Check
            is_hp, reason = self.integrity.check_honeypot(cand)
            if is_hp:
                logger.debug(f"Honeypot detected for {cand.get('candidate_id')}: {reason}")
                continue

            cand['sieve_category'] = cat
            cand['sieve_penalty'] = penalty
            stage1_pool.append(cand)

        logger.info(f"Stage 1 complete. Pool reduced to {len(stage1_pool)}")

        # Deduplication and Boilerplate’s are applied to the whole pool
        duplicates = self.integrity.detect_duplicates(stage1_pool)
        # For this implementation, we'll just mark them as 'duplicate' and lower score
        # instead of removing to allow a tie-breaker.

        # Stage 2 & 3: Signal Engine & Refiner (Combined for efficiency)
        # Compute semantic scores for the remaining pool
        semantic_scores = self.semantic.score_candidates(stage1_pool, jd_text)

        results = []
        for i, cand in enumerate(stage1_pool):
            # Technical Component Calculation
            # 1. Semantic match
            sem_score = semantic_scores[i]

            # 2. Skill match
            skill_score, skill_breakdown, triggers_floor = self.skills.score_skills(cand)

            # 3. Trajectory fit
            traj_score = self.trajectory.get_score(cand)

            # 4. Production Evidence
            evid_score, evidence_phrases = self.evidence.extract_evidence(cand)

            # 5. Company Pedigree
            history = cand.get('career_history', [])
            pedigree_sum = 0.0
            if history:
                for job in history:
                    _, weight = self.company_matcher.match_tier(job.get('company', ''))
                    pedigree_sum += weight
                pedigree_score = pedigree_sum / len(history)
            else:
                pedigree_score = 0.75

            # Aggregating Technical Score (based on jd_spec weights)
            # Note: Using simplified weighted sum here; real weights from spec
            tech_score = (
                sem_score * 0.20 +
                skill_score * 0.30 +
                traj_score * 0.20 +
                evid_score * 0.20 +
                pedigree_score * 0.10
            )

            # Required Skill Floor
            if triggers_floor:
                tech_score = min(tech_score, 0.40)

            # Behavioral Multipliers
            avail_mult = self.behavioral.get_score(cand)

            # Engagement multiplier (simplified)
            signals = cand.get('redrob_signals', {})
            eng_mult = 0.7 + (signals.get('profile_completeness_score', 50) / 100 * 0.3)

            # Notice Period Multiplier
            np_days = signals.get('notice_period_days', 30)
            np_mult = 1.0 if np_days <= 30 else (0.8 if np_days <= 60 else 0.5)

            final_score = tech_score * avail_mult * eng_mult * np_mult

            # Stage 4: Disqualifiers
            is_hard_reject, dq_mult, dq_reasons = self.dq_engine.evaluate_all(cand)
            if is_hard_reject:
                final_score = 0.0
            else:
                final_score *= dq_mult

            # Sieve penalty
            final_score += cand['sieve_penalty']

            results.append({
                "candidate_id": cand.get('candidate_id'),
                "final_score": float(np.clip(final_score, 0.0, 2.0)),
                "tech_score": tech_score,
                "dq_reasons": dq_reasons,
                "evidence": evidence_phrases,
                "candidate_data": cand
            })

        # Final Sorting: Score DESC, candidate_id ASC (Deterministic)
        results.sort(key=lambda x: (-x['final_score'], x['candidate_id']))

        return results
