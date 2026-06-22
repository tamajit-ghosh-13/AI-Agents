from typing import Dict, Any, List, Tuple
import json

class SkillMatcher:
    """
    Handles professional skill matching with proficiency and duration weights.
    Implements the 'two-required-skill-floor' rule.
    """
    def __init__(self, spec_path: str):
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        self.required = self.spec.get('required_skills', {})
        self.preferred = self.spec.get('preferred_skills', {})
        self.proficiency_map = {"expert": 1.2, "advanced": 1.0, "intermediate": 0.7, "beginner": 0.4}

    def score_skills(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float], bool]:
        """
        Computes a combined skill match score.
        Returns (total_score, skill_breakdown, triggers_floor_cap).
        """
        cand_skills = candidate.get('skills', [])
        if not cand_skills:
            return 0.0, {}, True if len(self.required) >= 2 else False

        # Map skills for easy lookup: {name_lower: {proficiency, duration}}
        skills_map = {s.get('name', '').lower(): s for s in cand_skills}

        skill_scores = {}
        required_met_count = 0

        # 1. Required Skills
        for skill_name, config in self.required.items():
            matched = False
            # Check skills array
            if skill_name.lower() in skills_map:
                s_obj = skills_map[skill_name.lower()]
                prof = s_obj.get('proficiency', 'intermediate').lower()
                mult = self.proficiency_map.get(prof, 0.5)

                # Duration bonus: > 2 years = full credit, < 1 year = partial
                duration = s_obj.get('duration_months', 0)
                dur_mult = 1.0 if duration >= 24 else (0.6 if duration >= 6 else 0.3)

                skill_scores[skill_name] = config['weight'] * mult * dur_mult
                matched = True

            # Fallback: Description mining (partial credit)
            if not matched:
                all_text = " ".join([j.get('description', '').lower() for j in candidate.get('career_history', [])]).lower()
                if any(form.lower() in all_text for form in config.get('surface_forms_skills_array', [])):
                    skill_scores[skill_name] = config['weight'] * 0.4 # 40% credit for mention without explicit skill object
                    matched = True
                elif any(form.lower() in all_text for form in config.get('surface_forms_description_mining', [])):
                    skill_scores[skill_name] = config['weight'] * 0.2 # 20% for semantic hint
                    matched = True

            if matched:
                required_met_count += 1

        # 2. Preferred Skills (Top 3 only)
        preferred_scores = []
        for skill_name, config in self.preferred.items():
            if skill_name.lower() in skills_map:
                s_obj = skills_map[skill_name.lower()]
                prof = s_obj.get('proficiency', 'intermediate').lower()
                mult = self.proficiency_map.get(prof, 0.5)
                preferred_scores.append(config['weight'] * mult)
            else:
                # Description mining for preferred
                all_text = " ".join([j.get('description', '').lower() for j in candidate.get('career_history', [])]).lower()
                if any(form.lower() in all_text for form in config.get('surface_forms', [])):
                    preferred_scores.append(config['weight'] * 0.3)

        # Take only top 3 preferred skills to prevent bloat gaming
        top_preferred_sum = sum(sorted(preferred_scores, reverse=True)[:3])

        # Normalize total
        total_weight = sum(self.required.values()) + sum(self.preferred.values()) # Simplified
        # Wait, weights in spec are absolute. Let's just sum them.

        final_score = sum(skill_scores.values()) + top_preferred_sum

        # Required skill floor: if < 2 required skills are met, cap total score later.
        # Here we just signal it.
        triggers_floor_cap = (required_met_count < 2)

        return float(final_score), skill_scores, triggers_floor_cap
