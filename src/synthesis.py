from typing import Dict, Any, List
import json
import yaml
import re


class ReasonSynthesizer:
    """
    Generates human-readable, evidence-based justifications for a candidate's rank.
    Dynamically extracts context, skills, and direct quotes of achievements from
    candidate resumes using keyword matching and a heuristic sentence priority scorer.
    """

    def synthesize(self, result: Dict[str, Any]) -> str:
        """
        Builds a highly personalized reasoning string based on the candidate's actual
        profile details. Achievement sentences are selected via a heuristic priority
        scorer that prefers quantified, production-evidence sentences.
        """
        cand = result['candidate_data']
        score = result['final_score']
        dq_reasons = result.get('key_risks', [])
        tier = result.get('tier', "unlikely_fit")

        if score == 0:
            return f"Rejected: {', '.join(dq_reasons) if dq_reasons else 'Does not meet basic requirements'}."

        # Load JD spec and company tiers
        try:
            with open("config/jd_spec.json", "r") as f:
                spec = json.load(f)
        except Exception:
            spec = {}

        try:
            with open("config/company_tiers.yaml", "r") as f:
                tiers_data = yaml.safe_load(f)
        except Exception:
            tiers_data = {}

        # 1. Candidate Info
        profile = cand.get('profile', {})
        exp = profile.get('years_of_experience', 0)
        title = profile.get('current_title', '')
        company = profile.get('current_company', '')
        location = profile.get('location', 'Unknown Location')
        notice = cand.get('redrob_signals', {}).get('notice_period_days', 30)

        history = cand.get('career_history', [])
        if not title and history:
            title = history[0].get('title', '')
        if not title:
            title = "AI / Software Engineer"
        if not company and history:
            company = history[0].get('company', '')

        if company:
            intro = f"This candidate is a {title} at {company} with {exp}y experience."
        else:
            intro = f"This candidate is a {title} with {exp}y experience."

        # 2. Company Pedigree from company_tiers.yaml
        ai_native_list = [c.lower() for c in tiers_data.get("ai_native", [])]
        tier1_list = [c.lower() for c in tiers_data.get("product_tier1", [])]
        product_list = (
            [c.lower() for c in tiers_data.get("product_tier2", [])]
            + [c.lower() for c in tiers_data.get("product_tier3_growth", [])]
            + [c.lower() for c in tiers_data.get("saas_b2b", [])]
        )

        aliases = {k.lower(): v.lower() for k, v in tiers_data.get("aliases", {}).items()}

        ai_companies = []
        tier1_companies = []
        product_companies = []

        for job in history:
            co = job.get('company', '')
            if not co:
                continue
            co_lower = co.lower()
            co_resolved = aliases.get(co_lower, co_lower)

            if any(ac == co_resolved or ac in co_resolved for ac in ai_native_list):
                ai_companies.append(co)
            elif any(t1 == co_resolved or t1 in co_resolved for t1 in tier1_list):
                tier1_companies.append(co)
            elif any(pr == co_resolved or pr in co_resolved for pr in product_list):
                product_companies.append(co)

        pedigree_parts = []
        if ai_companies:
            pedigree_parts.append(f"AI-native background at {ai_companies[0]}")
        if tier1_companies:
            pedigree_parts.append(f"Tier-1 engineering at {tier1_companies[0]}")
        if product_companies and not (ai_companies or tier1_companies):
            pedigree_parts.append(f"product-centric background at {product_companies[0]}")

        pedigree_str = ""
        if pedigree_parts:
            pedigree_str = f" They possess a strong {', and '.join(pedigree_parts)}."

        # 3. Required skills matched
        concrete_matches = []
        cand_skills = cand.get('skills', [])
        skills_map = {s.get('name', '').lower(): s for s in cand_skills}

        required_skills_config = spec.get('required_skills', {})
        for skill_cat, config in required_skills_config.items():
            if skill_cat.startswith('_'):
                continue
            surface_forms = config.get('surface_forms_skills_array', []) if isinstance(config, dict) else []
            for form in surface_forms:
                if form.lower() in skills_map:
                    concrete_matches.append(form)
                    break

        skills_str = ""
        if concrete_matches:
            unique_skills = []
            for s in concrete_matches:
                if s.lower() not in [us.lower() for us in unique_skills]:
                    unique_skills.append(s)
            skills_str = f" Technically, they bring verified skills in {', '.join(unique_skills[:4])}."

        # 4. Production Evidence & Metric-driven Achievements
        sentences = []
        for job in history:
            desc = job.get('description', '')
            if not desc:
                continue
            split_sentences = re.split(r'\.\s+', desc)
            for s in split_sentences:
                s = s.strip()
                if not s:
                    continue
                s_lower = s.lower()
                keywords = [
                    "production", "ndcg", "mrr", "map", "re-ranker", "reranker",
                    "embedding", "vector", "search", "shipped", "deployed",
                    "implemented", "scaled", "latency", "pipeline", "built", "owned",
                ]
                if any(kw in s_lower for kw in keywords):
                    s_clean = re.sub(r'\s+', ' ', s)
                    if s_clean.endswith('.'):
                        s_clean = s_clean[:-1]
                    if s_clean and 25 < len(s_clean) < 180:
                        sentences.append(s_clean)

        def sentence_priority(s: str) -> int:
            score = 0
            s_lower = s.lower()
            if "%" in s or any(char.isdigit() for char in s):
                score += 3
            if "production" in s_lower or "shipped" in s_lower or "deployed" in s_lower:
                score += 2
            if (
                "ndcg" in s_lower
                or "mrr" in s_lower
                or "re-ranker" in s_lower
                or "vector" in s_lower
                or "embedding" in s_lower
            ):
                score += 2
            return score

        best_sentences = sorted(sentences, key=sentence_priority, reverse=True)
        achievements_str = ""
        if best_sentences:
            unique_achievements = []
            for a in best_sentences:
                if not any(ua[:15].lower() == a[:15].lower() for ua in unique_achievements):
                    unique_achievements.append(a)
                if len(unique_achievements) >= 2:
                    break

            quoted = [f'"{a}"' for a in unique_achievements]
            achievements_str = f" Key achievements from their history include: {'; and '.join(quoted)}."

        # 5. Notice Period & Location
        availability_str = f" They are based in {location} (notice period: {notice} days)."

        # Assemble full reasoning
        reasoning = f"[{tier}] {intro}{pedigree_str}{skills_str}{achievements_str}{availability_str}"
        reasoning = re.sub(r'\.+', '.', reasoning)

        if dq_reasons:
            reasoning += f" Note: {', '.join(dq_reasons)}."

        return reasoning
