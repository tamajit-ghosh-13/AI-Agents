import yaml
from loguru import logger
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple, Set

class IntegrityGuard:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.service_companies = set(self.config.get('service_companies', []))

        # Ownership verbs that indicate a "Shipper" profile
        self.ownership_verbs = {
            "shipped", "owned", "built", "led", "designed", "deployed",
            "scaled", "reduced", "increased", "optimized", "implemented",
            "architected", "delivered", "engineered"
        }

    def check_honeypot(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        
        # Detects impossible profiles that are clearly synthetic honeypots.
        
        # 1. Experience vs Job Duration Contradiction
        total_exp = candidate.get('profile', {}).get('years_of_experience', 0)
        history = candidate.get('career_history', [])
        calculated_exp = sum(job.get('duration_months', 0) for job in history) / 12

        if total_exp > 0 and abs(total_exp - calculated_exp) > 1.5:
            return True, f"Experience mismatch: Profile says {total_exp}y, history shows {calculated_exp:.1f}y"

        # 2. Expert proficiency with minimal duration
        skills = candidate.get('skills', [])
        for skill in skills:
            if skill.get('proficiency') == 'expert' and (skill.get('duration_months', 0) < 6):
                return True, f"Impossible skill: Expert in {skill.get('name')} with < 6 months experience"

        # 3. Profile completeness vs content (Empty but high scores)
        signals = candidate.get('redrob_signals', {})
        if signals.get('profile_completeness_score', 100) < 30 and not history:
            return True, "Empty profile with incongruent signals"

        return False, ""

    def detect_boilerplate(self, descriptions: List[str]) -> List[int]:
        
        # Flags candidates whose descriptions are identical or near-identical to others, suggesting copy-pasted templates.
        
        if not descriptions:
            return []

        # Simple Jaccard Similarity for boilerplate detection
        def get_set(text):
            return set(text.lower().split())

        flagged = []
        for i in range(len(descriptions)):
            set_i = get_set(descriptions[i])
            if not set_i: continue
            for j in range(i + 1, len(descriptions)):
                set_j = get_set(descriptions[j])
                if not set_j: continue

                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                if intersection / union > 0.8: # 80% similarity
                    flagged.append(i)
                    flagged.append(j)

        return list(set(flagged))

    def evaluate_pedigree(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        
        # Implements the Evidence-Based Pedigree Filter.
        # Instead of blacklisting service companies, we check for 'ownership' signals.
        
        history = candidate.get('career_history', [])
        if not history:
            return 1.0, "No history to evaluate"

        all_companies = [job.get('company', '').upper() for job in history]
        is_pure_service = all(any(svc.upper() in comp for svc in self.service_companies) for comp in all_companies)

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
        
        # Detects near-duplicate candidates based on profile and history.
        
        seen_hashes = {}
        duplicates = set()

        for cand in candidates:
            # Create a simplified fingerprint of the candidate
            fingerprint = (
                cand.get('profile', {}).get('summary', '').strip().lower(),
                tuple(sorted([job.get('company', '') for job in cand.get('career_history', [])]))
            )
            if fingerprint in seen_hashes:
                duplicates.add(cand['candidate_id'])
                duplicates.add(seen_hashes[fingerprint])
            else:
                seen_hashes[fingerprint] = cand['candidate_id']

        return duplicates
