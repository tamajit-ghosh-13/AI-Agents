import yaml
from loguru import logger
from typing import List, Dict, Any, Tuple, Set
import re

class IntegrityGuard:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Canonicalize service companies from config
        self.service_companies = set(c.lower() for c in self.config.get('service_companies', []))

        # Ownership verbs that indicate a "Shipper" profile
        self.ownership_verbs = {
            "shipped", "owned", "built", "led", "designed", "deployed",
            "scaled", "reduced", "increased", "optimized", "implemented",
            "architected", "delivered", "engineered"
        }

    def check_honeypot(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """Detects synthetic honeypots via data contradictions."""
        # 1. Experience vs Job Duration Contradiction
        # Use a more lenient tolerance (e.g. 1 year) and account for overlapping roles
        total_exp = candidate.get('profile', {}).get('years_of_experience', 0)
        history = candidate.get('career_history', [])

        # Calculate unique months of experience to handle overlaps
        if not history:
            calculated_exp = 0
        else:
            # This is a simplification; real gap-aware analysis happens in trajectory.py
            calculated_exp = sum(job.get('duration_months', 0) for job in history) / 12

        # Lenient check: only flag if discrepancy is massive (> 2 years)
        if total_exp > 0 and abs(total_exp - calculated_exp) > 2.0:
            return True, f"Experience mismatch: Profile {total_exp}y, History {calculated_exp:.1f}y"

        # 2. Expert proficiency with minimal duration
        # Allow for transferable skills (e.g. an expert Python dev learning Go)
        skills = candidate.get('skills', [])
        for skill in skills:
            if skill.get('proficiency') == 'expert':
                duration = skill.get('duration_months', 0)
                # Only flag if TOTAL career experience is also very low
                if duration < 6 and total_exp < 1:
                    return True, f"Impossible skill: Expert in {skill.get('name')} with minimal career exp"

        # 3. Profile completeness vs content
        signals = candidate.get('redrob_signals', {})
        if signals.get('profile_completeness_score', 100) < 30 and not history:
            return True, "Empty profile with incongruent signals"

        return False, ""

    def detect_boilerplate(self, candidates: List[Dict[str, Any]]) -> Set[str]:
        """
        Flags candidates with near-identical descriptions.
        Returns a set of candidate IDs.
        """
        descriptions = []
        cand_ids = []
        for c in candidates:
            cid = c.get('candidate_id')
            if not cid: continue

            # Combine all job descriptions for the candidate
            text = " ".join([j.get('description', '').lower() for j in c.get('career_history', [])]).strip()
            if text:
                descriptions.append(text)
                cand_ids.append(cid)

        if not descriptions:
            return set()

        flagged = set()
        # Optimized similarity check: only compare if length is similar
        for i in range(len(descriptions)):
            set_i = set(descriptions[i].split())
            if not set_i: continue
            for j in range(i + 1, len(descriptions)):
                set_j = set(descriptions[j].split())
                if not set_j: continue

                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                if intersection / union > 0.85: # Higher threshold for boilerplate
                    flagged.add(cand_ids[i])
                    flagged.add(cand_ids[j])

        return flagged

    def evaluate_pedigree(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        """Evidence-Based Pedigree Filter."""
        history = candidate.get('career_history', [])
        if not history:
            return 1.0, "No history to evaluate"

        # Exact/Tokenized match instead of substring to avoid "CS" matching "Microsoft"
        is_pure_service = True
        for job in history:
            comp = job.get('company', '').lower().strip()
            # If company is NOT in service list, it's not a pure service background
            if comp not in self.service_companies:
                is_pure_service = False
                break

        if not is_pure_service:
            return 1.0, "Diverse or product-based pedigree"

        # Candidate is from pure service background. Check for "Ownership Verbs".
        has_ownership_signal = False
        for job in history:
            desc = job.get('description', '').lower()
            if any(verb in desc for verb in self.ownership_verbs):
                has_ownership_signal = True
                break

        if has_ownership_signal:
            return 1.0, "Service background but demonstrates ownership signals"
        else:
            return 0.7, "Pure service background without clear ownership evidence"

    def detect_duplicates(self, candidates: List[Dict[str, Any]]) -> Set[str]:
        """Detects near-duplicate candidates via stronger fingerprinting."""
        seen_hashes = {}
        duplicates = set()

        for cand in candidates:
            cid = cand.get('candidate_id')
            if not cid: continue

            # Fingerprint: Normalized summary + Sorted list of companies + total exp
            fingerprint = (
                cand.get('profile', {}).get('summary', '').strip().lower(),
                tuple(sorted([job.get('company', '').lower().strip() for job in cand.get('career_history', [])])),
                cand.get('profile', {}).get('years_of_experience', 0)
            )

            if fingerprint in seen_hashes:
                duplicates.add(cid)
                duplicates.add(seen_hashes[fingerprint])
            else:
                seen_hashes[fingerprint] = cid

        return duplicates
