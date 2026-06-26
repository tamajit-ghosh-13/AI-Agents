from typing import Dict, Any, Tuple
import yaml

class CompanyMatcher:
    """
    Normalizes company names and maps them to tiers from company_tiers.yaml.
    """
    def __init__(self, tiers_path: str):
        with open(tiers_path, 'r') as f:
            self.tiers_data = yaml.safe_load(f)

        self.aliases = self.tiers_data.get('aliases', {})
        self.tier_weights = self.tiers_data.get('tier_weights', {})
        self.canonical_map = {}

        # Build a reverse map: {canonical_name_lower: tier}
        for tier, companies in self.tiers_data.items():
            if isinstance(companies, list):
                for co in companies:
                    self.canonical_map[co.lower()] = tier

    def normalize_name(self, name: str) -> str:
        if not name: return ""
        # Strip common suffixes
        name = name.replace(" Ltd.", "").replace(" Pvt Ltd", "").replace(" Limited", "").replace(" Inc.", "")
        name = name.strip()
        return name

from typing import Dict, Any, Tuple, List
import yaml
from src.orchestration.types import Verdict, Evidence

class CompanyMatcher:
    """
    Normalizes company names and maps them to tiers from company_tiers.yaml.
    """
    def __init__(self, tiers_path: str):
        with open(tiers_path, 'r') as f:
            self.tiers_data = yaml.safe_load(f)

        self.aliases = self.tiers_data.get('aliases', {})
        self.tier_weights = self.tiers_data.get('tier_weights', {})
        self.canonical_map = {}

        # Build a reverse map: {canonical_name_lower: tier}
        for tier, companies in self.tiers_data.items():
            if isinstance(companies, list):
                for co in companies:
                    self.canonical_map[co.lower()] = tier

    def normalize_name(self, name: str) -> str:
        if not name: return ""
        name = name.replace(" Ltd.", "").replace(" Pvt Ltd", "").replace(" Limited", "").replace(" Inc.", "")
        name = name.strip()
        return name

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Evaluates the overall company pedigree of the candidate's career history.
        """
        history = candidate.get('career_history', [])
        if not history:
            return Verdict(
                agent="ProductContextAgent",
                signal="none",
                confidence=1.0,
                reasoning="No career history provided.",
                score=0.75
            )

        total_weight = 0.0
        evidence_list = []

        for job in history:
            company_name = job.get('company', '')
            tier, weight = self._match_tier(company_name)
            total_weight += weight
            if tier != "unclassified":
                evidence_list.append(Evidence(text=f"Worked at {company_name} ({tier})", source="career_history"))

        avg_weight = total_weight / len(history)
        normalized_score = min(1.0, avg_weight)

        if normalized_score > 0.9: signal = "strong"
        elif normalized_score > 0.7: signal = "moderate"
        elif normalized_score > 0.4: signal = "weak"
        else: signal = "none"

        return Verdict(
            agent="ProductContextAgent",
            signal=signal,
            confidence=0.9,
            evidence=evidence_list,
            reasoning=f"Average company pedigree score: {normalized_score:.2f} based on {len(history)} roles.",
            score=normalized_score
        )

    def _match_tier(self, company_name: str) -> Tuple[str, float]:
        norm_name = self.normalize_name(company_name)
        if not norm_name:
            return "unclassified", self.tier_weights.get("unclassified", 0.75)

        for canonical, alias in self.aliases.items():
            if alias.lower() == norm_name.lower():
                tier = self.canonical_map.get(canonical.lower(), "unclassified")
                return tier, self.tier_weights.get(tier, 0.75)

        norm_lower = norm_name.lower()
        if norm_lower in self.canonical_map:
            tier = self.canonical_map[norm_lower]
            return tier, self.tier_weights.get(tier, 0.75)

        for canonical, tier in self.canonical_map.items():
            if len(canonical) > 3 and canonical in norm_lower:
                return tier, self.tier_weights.get(tier, 0.75)

        return "unclassified", self.tier_weights.get("unclassified", 0.75)
