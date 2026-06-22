from typing import Dict, Any, Tuple, List
import json
from datetime import datetime

class DisqualifierEngine:
    """
    Implements DQ1-DQ7 as explicit rule modules.
    These run BEFORE final score aggregation.
    """
    def __init__(self, spec_path: str, company_tiers_path: str):
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        # Load company tiers for DQ5 and DQ7
        with open(company_tiers_path, 'r') as f:
            import yaml
            self.tiers = yaml.safe_load(f)

        self.dq_rules = self.spec.get('disqualifier_rules', {})

    def _get_product_companies(self) -> set:
        # Combine all product-oriented tiers from company_tiers.yaml
        product_tiers = [
            'product_tier1', 'ai_native', 'product_tier2',
            'product_tier3_growth', 'saas_b2b', 'gcc_captive'
        ]
        companies = set()
        for tier in product_tiers:
            companies.update(self.tiers.get(tier, []))
        return companies

    def evaluate_all(self, candidate: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """
        Evaluates all DQ rules.
        Returns (is_hard_reject, total_multiplier, triggered_rules).
        """
        is_hard_reject = False
        total_multiplier = 1.0
        triggered_rules = []

        # Run DQ1 to DQ7
        for dq_id in ["DQ1_pure_research_career", "DQ2_shallow_ai_experience", "DQ3_non_coding_senior",
                      "DQ4_title_inflation_hopper", "DQ5_pure_consulting_background",
                      "DQ6_wrong_primary_domain", "DQ7_closed_source_only"]:

            triggered, multiplier, reason = self._check_rule(dq_id, candidate)
            if triggered:
                triggered_rules.append(f"{dq_id}: {reason}")
                if self.dq_rules.get(dq_id, {}).get('severity') == 'hard_reject':
                    is_hard_reject = True
                    total_multiplier = 0.0
                    break # Hard reject immediately
                else:
                    total_multiplier *= multiplier

        return is_hard_reject, total_multiplier, triggered_rules

    def _check_rule(self, rule_id: str, candidate: Dict[str, Any]) -> Tuple[bool, float, str]:
        rule = self.dq_rules.get(rule_id, {})
        multiplier = rule.get('score_multiplier', 1.0)

        if rule_id == "DQ1_pure_research_career":
            # DQ1: Pure research career
            history = candidate.get('career_history', [])
            if not history: return False, 1.0, ""

            # Signal A: All titles contain Research/Researcher/Scientist or University/Lab
            research_titles = {'research', 'researcher', 'scientist'}
            titles_all_research = all(
                any(kw in job.get('title', '').lower() for kw in research_titles) or
                self._is_academic_employer(job.get('company', ''))
                for job in history
            )

            # Signal B: Zero production-evidence verbs
            prod_verbs = {"deployed", "shipped", "production", "launched", "served", "live system"}
            all_descs = " ".join([job.get('description', '').lower() for job in history])
            no_prod_evidence = not any(verb in all_descs for verb in prod_verbs)

            # Signal C: All companies are academic/research
            all_academic = all(self._is_academic_employer(job.get('company', '')) for job in history)

            if titles_all_research and no_prod_evidence and all_academic:
                return True, multiplier, "Pure research career without production evidence"

        elif rule_id == "DQ2_shallow_ai_experience":
            # DQ2: Shallow AI experience
            history = candidate.get('career_history', [])
            if not history: return False, 1.0, ""

            # Signal A: AI la after 2022-06-01
            ai_start_date_cutoff = datetime(2022, 6, 1)
            ai_roles = [job for job in history if any(kw in job.get('description', '').lower() for kw in ["ml", "ai", "llm"])]
            all_recent = True
            for role in ai_roles:
                start_date = role.get('start_date')
                if start_date:
                    try:
                        if datetime.strptime(start_date, '%Y-%m-%d') < ai_start_date_cutoff:
                            all_recent = False
                            break
                    except ValueError: pass

            if not ai_roles: all_recent = False # Must have some AI experience to be "shallow"

            # Signal B: Only wrappers
            wrappers = {"langchain", "chatgpt api", "openai api", "gpt-4"}
            all_descs = " ".join([job.get('description', '').lower() for job in history])
            only_wrappers = all(any(w in all_descs for w in wrappers) for _ in [1]) # Simplified check

            # Signal C: No pre-LLM production ML
            pre_llm_signals = {"recommendation engine", "ranking system", "search pipeline", "embedding model", "ndcg", "mrr"}
            no_pre_llm = not any(s in all_descs for s in pre_llm_signals)

            if all_recent and only_wrappers and no_pre_llm:
                return True, multiplier, "Shallow AI experience (wrappers only, no depth)"

        elif rule_id == "DQ3_non_coding_senior":
            # DQ3: Non-coding senior
            history = candidate.get('career_history', [])
            if not history: return False, 1.0, ""
            recent_role = history[0]
            title = recent_role.get('title', '').lower()
            senior_titles = {"principal engineer", "distinguished engineer", "staff engineer", "engineering manager", "vp engineering", "chief architect", "head of engineering", "cto"}

            if any(st in title for st in senior_titles):
                desc = recent_role.get('description', '').lower()
                leadership_verbs = {"led", "defined", "mentored", "planned", "strategized", "reviewed", "managed"}
                tech_verbs = {"built", "implemented", "wrote", "coded", "shipped", "deployed", "trained", "fine-tuned"}

                has_leadership = any(v in desc for v in leadership_verbs)
                has_tech = any(v in desc for v in tech_verbs)

                if has_leadership and not has_tech:
                    return True, multiplier, "Senior role with no individual technical contribution evidence"

        elif rule_id == "DQ4_title_inflation_hopper":
            # DQ4: Title inflation hopper
            history = candidate.get('career_history', [])
            hierarchy = self.spec['disqualifier_rules']['DQ4_title_inflation_hopper']['title_hierarchy']

            promotions = 0
            for i in range(len(history)-1):
                t1 = history[i+1].get('title', '') # Prior role
                t2 = history[i].get('title', '')   # Current/Later role

                # Check if t2 is higher in hierarchy than t1
                try:
                    idx1 = next(i for i, v in enumerate(hierarchy) if v.lower() in t1.lower())
                    idx2 = next(i for i, v in enumerate(hierarchy) if v.lower() in t2.lower())
                    if idx2 > idx1 and history[i].get('duration_months', 0) < 18:
                        promotions += 1
                except StopIteration:
                    continue

            if promotions >= 2:
                return True, multiplier, "Title inflation detected: multiple rapid promotions < 18mo"

        elif rule_id == "DQ5_pure_consulting_background":
            # DQ5: Pure consulting
            history = candidate.get('career_history', [])
            if not history: return False, 1.0, ""

            product_cos = self._get_product_companies()
            all_consulting = True
            has_product_exp = False

            for job in history:
                comp = job.get('company', '')
                if comp in product_cos:
                    has_product_exp = True
                    all_consulting = False
                    break

                # Check for ownership in consulting role
                desc = job.get('description', '').lower()
                ownership = {"shipped", "owned end-to-end", "built from scratch", "deployed to production"}
                if any(o in desc for o in ownership):
                    has_product_exp = True
                    all_consulting = False
                    break

            if all_consulting and not has_product_exp:
                return True, multiplier, "Pure consulting background without product experience"

        elif rule_id == "DQ6_wrong_primary_domain":
            # DQ6: Wrong primary domain
            all_text = (candidate.get('profile', {}).get('summary', '') + " " +
                        " ".join([j.get('description', '') for j in candidate.get('career_history', [])])).lower()

            cv_signals = self.dq_rules['DQ6_wrong_primary_domain']['cv_speech_robotics_signals']
            nlp_signals = self.dq_rules['DQ6_wrong_primary_domain']['nlp_ir_salvage_signals']

            cv_count = sum(1 for s in cv_signals if s.lower() in all_text)
            nlp_count = sum(1 for s in nlp_signals if s.lower() in all_text)

            if cv_count > 0 and nlp_count == 0:
                # Simple ratio: if CV is dominant and NLP is zero
                return True, multiplier, "Primary expertise in CV/Speech/Robotics with no NLP/IR evidence"

        elif rule_id == "DQ7_closed_source_only":
            # DQ7: Closed source only
            if candidate.get('profile', {}).get('years_of_experience', 0) < 5:
                return False, 1.0, ""

            signals = candidate.get('redrob_signals', {})
            github_score = signals.get('github_activity_score', -1)
            certs = candidate.get('certifications', [])

            all_text = (candidate.get('profile', {}).get('summary', '') + " " +
                        " ".join([j.get('description', '') for j in candidate.get('career_history', [])])).lower()
            oss_keywords = {"paper", "publication", "open source", "oss", "github", "conference", "blog", "talk", "speaker", "arxiv"}

            has_oss = any(k in all_text for k in oss_keywords)

            if (github_score <= 10) and (len(certs) < 2) and not has_oss:
                return True, multiplier, "Low external validation for 5+ year experience"

        return False, 1.0, ""

    def _is_academic_employer(self, company: str) -> bool:
        if not company: return False
        company = company.lower()
        academic_keywords = {"university", "institute", "college", "lab", "national laboratory", "research center"}
        return any(kw in company for kw in academic_keywords)
