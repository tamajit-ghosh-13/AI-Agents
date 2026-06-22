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
            if skill_name.startswith('_'): continue
            matched = False
            # Check skills array
            if skill_name.lower() in skills_map:
                s_obj = skills_map[skill_name.lower()]
                prof = s_obj.get('proficiency', 'intermediate').lower()
                mult = self.proficiency_map.get(prof, 0.5)

                # Duration bonus: > 2 years = full credit, < 1 year = partial
                duration = s_obj.get('duration_months', 0)
                dur_mult = 1.0 if duration >= 24 else (0.6 if duration >= 6 else 0.3)

                weight = config.get('weight', 1.0) if isinstance(config, dict) else 1.0
                skill_scores[skill_name] = weight * mult * dur_mult
                matched = True

            # Fallback: Description mining (partial credit)
            if not matched:
                all_text = " ".join([j.get('description', '').lower() for j in candidate.get('career_history', [])]).lower()
                if isinstance(config, dict) and any(form.lower() in all_text for form in config.get('surface_forms_skills_array', [])):
                    weight = config.get('weight', 1.0)
                    skill_scores[skill_name] = weight * 0.4 # 40% credit for mention without explicit skill object
                    matched = True
                elif isinstance(config, dict) and any(form.lower() in all_text for form in config.get('surface_forms_description_mining', [])):
                    weight = config.get('weight', 1.0)
                    skill_scores[skill_name] = weight * 0.2 # 20% for semantic hint
                    matched = True

            if matched:
                required_met_count += 1

        # 2. Preferred Skills (Top 3 only)
        preferred_scores = []
        for skill_name, config in self.preferred.items():
            if skill_name.startswith('_'): continue
            if skill_name.lower() in skills_map:
                s_obj = skills_map[skill_name.lower()]
                prof = s_obj.get('proficiency', 'intermediate').lower()
                mult = self.proficiency_map.get(prof, 0.5)
                weight = config.get('weight', 0.4) if isinstance(config, dict) else 0.4
                preferred_scores.append(weight * mult)
            else:
                # Description mining for preferred
                all_text = " ".join([j.get('description', '').lower() for j in candidate.get('career_history', [])]).lower()
                surface_forms = config.get('surface_forms', []) if isinstance(config, dict) else []
                if any(form.lower() in all_text for form in surface_forms):
                    weight = config.get('weight', 0.4) if isinstance(config, dict) else 0.4
                    preferred_scores.append(weight * 0.3)

        # Take only top 3 preferred skills to prevent bloat gaming
        top_preferred_sum = sum(sorted(preferred_scores, reverse=True)[:3])

        # Normalize total
        # Extract weights from the config dictionaries
        required_weights = [config.get('weight', 1.0) if isinstance(config, dict) else 1.0 for config in self.required.values()]
        preferred_weights = [config.get('weight', 0.4) if isinstance(config, dict) else 0.4 for config in self.preferred.values()]

        total_weight = sum(required_weights) + sum(preferred_weights)

        final_score = sum(skill_scores.values()) + top_preferred_sum

        # Required skill floor: if < 2 required skills are met, cap total score later.
        # Here we just signal it.
        triggers_floor_cap = (required_met_count < 2)

        return float(final_score), skill_scores, triggers_floor_cap
