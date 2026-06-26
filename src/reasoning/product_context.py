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

    def match_tier(self, company_name: str) -> Tuple[str, float]:
        """
        Returns (tier_name, weight).
        """
        norm_name = self.normalize_name(company_name)
        if not norm_name:
            return "unclassified", self.tier_weights.get("unclassified", 0.75)

        # 1. Check aliases first
        for canonical, alias in self.aliases.items():
            if alias.lower() == norm_name.lower():
                return self.canonical_map.get(canonical.lower(), "unclassified"), self.tier_weights.get(self.canonical_map.get(canonical.lower(), "unclassified"), 0.75)

        # 2. Exact match
        norm_lower = norm_name.lower()
        if norm_lower in self.canonical_map:
            tier = self.canonical_map[norm_lower]
            return tier, self.tier_weights.get(tier, 0.75)

        # 3. Substring containment (Last resort, careful with "CS" -> "Microsoft")
        # Only match if the normalized name is at least 3 chars and we have a good hit
        for canonical, tier in self.canonical_map.items():
            if len(canonical) > 3 and canonical in norm_lower:
                return tier, self.tier_weights.get(tier, 0.75)

        return "unclassified", self.tier_weights.get("unclassified", 0.75)
