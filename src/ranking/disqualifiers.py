import re
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime, timedelta
from src.orchestration.types import Verdict, Evidence


def _kw_in(text: str, keyword: str) -> bool:
    """
    Word-boundary-safe keyword search.
    Plain substring checks (e.g. "ai" in text, "lab" in text, "oss" in text)
    false-positive constantly: "ai" matches inside "domain"/"maintain"/"claim",
    "lab" matches inside "collaborate", "oss" matches inside "across"/"boss".
    This applies \b on both sides, which also works for multi-word phrases
    like "open source".
    """
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return re.search(pattern, text) is not None


def _any_kw(text: str, keywords) -> bool:
    return any(_kw_in(text, kw) for kw in keywords)


def _all_kw_absent(text: str, keywords) -> bool:
    return not _any_kw(text, keywords)


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

        # DQ7 ("closed-source only, no external validation") is listed in the
        # JD under its softer "things we explicitly do NOT want" framing, not
        # under the explicit disqualifiers it says it will hard-reject for.
        # It's also inherently noisy -- a recruiting platform profile rarely
        # captures papers/talks/OSS reliably, and missing GitHub data is not
        # the same thing as a confirmed lack of external validation. So this
        # rule is never allowed to hard-reject, even if the spec file marks
        # it that way -- it only ever applies a mild score multiplier.
        self._never_hard_reject = {"DQ7_closed_source_only"}

        # Explicit consulting-firm list per JD: "TCS, Infosys, Wipro, Accenture,
        # Cognizant, Capgemini, etc." Matching against this directly (rather than
        # "not classified as a product company") avoids silently flagging
        # legitimate product companies that simply aren't in company_tiers.yaml.
        self._consulting_firms = {
            "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
            "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
            "ibm global services", "mindtree", "mphasis", "l&t infotech", "ltimindtree",
        }

        # JD's named "wrong primary domain" categories and the NLP/IR signal
        # that, if present anywhere in the career history, should exempt the
        # candidate (the JD's bar is "without significant NLP/IR exposure" --
        # not "ever mentioned a CV term once").
        self._cv_speech_robotics_kw = {
            "computer vision", "image classification", "image moderation",
            "object detection", "resnet", "cnn", "convolutional neural network",
            "speech recognition", "asr", "text-to-speech", "tts",
            "robotics", "slam", "ros", "autonomous vehicle", "lidar",
        }
        self._nlp_ir_kw = {
            "nlp", "natural language processing", "semantic search", "embedding",
            "embeddings", "retrieval", "ranking system", "ranking model",
            "recommendation engine", "recommendation system", "information retrieval",
            "ndcg", "mrr", "map@", "rag", "vector database", "vector search",
            "search pipeline", "llm",
        }

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

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Evaluates all DQ rules.
        Returns a structured Verdict.
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
                triggered_rules.append(reason)
                severity = self.dq_rules.get(dq_id, {}).get('severity', '')
                # Three label strings in jd_spec.json: 'hard_reject' (DQ1),
                # 'soft_reject' (DQ2-5), and 'reject' (DQ6, DQ7).
                # Only 'hard_reject' zeroes the candidate; 'reject' and 'soft_reject'
                # both route to the soft multiplier path.
                # DQ7 is additionally protected by _never_hard_reject, which prevents
                # it from zeroing even if the spec file were to label it 'hard_reject'.
                rule_is_hard_reject = severity == 'hard_reject'
                rule_is_soft = severity in ('soft_reject', 'reject')

                if rule_is_hard_reject and dq_id not in self._never_hard_reject:
                    is_hard_reject = True
                    total_multiplier = 0.0
                    break  # Hard reject immediately
                elif rule_is_hard_reject and dq_id in self._never_hard_reject:
                    # Spec says hard_reject, but this rule is never allowed to
                    # zero the candidate out -- apply a mild penalty instead.
                    # A floor keeps it meaningfully "mild" even if a stale
                    # spec value were something aggressive like 0.0 or 0.1.
                    mild_multiplier = max(multiplier, 0.6)
                    total_multiplier *= mild_multiplier
                else:
                    # Covers both soft_reject and reject -- always a multiplier,
                    # never a zero.  DQ6 ('reject', multiplier=0.2) stays recoverable.
                    total_multiplier *= multiplier

        signal = "none" if is_hard_reject else ("moderate" if len(triggered_rules) == 0 else "weak")
        score = 0.0 if is_hard_reject else total_multiplier

        return Verdict(
            agent="DisqualifierEngine",
            signal=signal,
            confidence=1.0,
            evidence=[Evidence(text=r, source="disqualifier_rules") for r in triggered_rules],
            risks=triggered_rules,
            reasoning=f"Triggered {len(triggered_rules)} disqualifier rules. Hard reject: {is_hard_reject}.",
            score=score
        )

    def _check_rule(self, rule_id: str, candidate: Dict[str, Any]) -> Tuple[bool, float, str]:
        rule = self.dq_rules.get(rule_id, {})
        multiplier = rule.get('score_multiplier', 1.0)

        if rule_id == "DQ1_pure_research_career":
            # DQ1: Pure research career, no production deployment ever.
            history = candidate.get('career_history', [])
            if not history:
                return False, 1.0, ""

            research_titles = {'research', 'researcher', 'scientist', 'phd', 'postdoc', 'post-doc'}
            titles_all_research = all(
                _any_kw(job.get('title', '').lower(), research_titles) or
                self._is_academic_employer(job.get('company', ''))
                for job in history
            )

            prod_verbs = {"deployed", "shipped", "production", "launched", "served", "live system"}
            all_descs = " ".join([job.get('description', '').lower() for job in history])
            no_prod_evidence = _all_kw_absent(all_descs, prod_verbs)

            all_academic = all(self._is_academic_employer(job.get('company', '')) for job in history)

            if titles_all_research and no_prod_evidence and all_academic:
                return True, multiplier, "Pure research career without production evidence"

        elif rule_id == "DQ2_shallow_ai_experience":
            # DQ2: Shallow AI experience -- JD's bar is "recent (under 12 months)
            # projects using LangChain to call OpenAI" with no pre-LLM-era
            # production ML depth.
            history = candidate.get('career_history', [])
            if not history:
                return False, 1.0, ""

            # Recency cutoff must be relative to *today*, not a fixed historical
            # date -- otherwise this rule silently stops matching the JD's
            # "under 12 months" framing as time passes.
            ai_recency_cutoff = datetime.now() - timedelta(days=365)

            ai_keywords = {"ml", "ai", "llm"}
            ai_roles = [job for job in history if _any_kw(job.get('description', '').lower(), ai_keywords)]

            if not ai_roles:
                return False, 1.0, ""  # Must have some AI experience to be "shallow"

            all_recent = True
            for role in ai_roles:
                start_date = role.get('start_date')
                if start_date:
                    try:
                        if datetime.strptime(start_date, '%Y-%m-%d') < ai_recency_cutoff:
                            all_recent = False
                            break
                    except ValueError:
                        pass

            # "Only wrappers": wrapper keywords present AND no signal of any
            # substantial, non-wrapper ML tooling/depth anywhere in the history.
            # (The original implementation only checked "any wrapper keyword
            # present," which flagged candidates with real depth who merely
            # mentioned LangChain in passing.)
            wrappers = {"langchain", "chatgpt api", "openai api", "gpt-4"}
            all_descs = " ".join([job.get('description', '').lower() for job in history])
            has_wrapper_signal = _any_kw(all_descs, wrappers)

            substantial_ml_signals = {
                "fine-tun", "fine tun", "training pipeline", "pytorch", "tensorflow",
                "embedding model", "vector database", "from scratch", "lora", "qlora", "peft",
            }
            no_substantial_depth = _all_kw_absent(all_descs, substantial_ml_signals)

            pre_llm_signals = {
                "recommendation engine", "ranking system", "search pipeline",
                "embedding model", "ndcg", "mrr",
            }
            no_pre_llm = _all_kw_absent(all_descs, pre_llm_signals)

            if all_recent and has_wrapper_signal and no_substantial_depth and no_pre_llm:
                return True, multiplier, "Shallow AI experience (wrappers only, no depth)"

        elif rule_id == "DQ3_non_coding_senior":
            # DQ3: Senior who hasn't written production code in the last 18
            # months (moved into pure architecture/tech-lead). The JD's window
            # is 18 months -- check every role that falls inside it, not just
            # the single most recent job.
            history = candidate.get('career_history', [])
            if not history:
                return False, 1.0, ""

            senior_titles = {
                "principal engineer", "distinguished engineer", "staff engineer",
                "engineering manager", "vp engineering", "chief architect",
                "head of engineering", "cto", "tech lead", "technical lead",
                "engineering lead", "senior staff engineer",
            }

            cutoff = datetime.now() - timedelta(days=18 * 30)
            recent_roles = []
            for job in history:
                start_date = job.get('start_date')
                if start_date:
                    try:
                        if datetime.strptime(start_date, '%Y-%m-%d') >= cutoff:
                            recent_roles.append(job)
                            continue
                    except ValueError:
                        pass
                # If we can't parse a date, fall back to treating only the
                # most recent role (history[0]) as "recent" rather than
                # silently dropping the candidate from this check.
                if job is history[0]:
                    recent_roles.append(job)

            if not recent_roles:
                return False, 1.0, ""

            is_senior_recently = any(
                _any_kw(job.get('title', '').lower(), senior_titles) for job in recent_roles
            )
            if not is_senior_recently:
                return False, 1.0, ""

            leadership_verbs = {"led", "defined", "mentored", "planned", "strategized", "reviewed", "managed"}
            tech_verbs = {"built", "implemented", "wrote", "coded", "shipped", "deployed", "trained", "fine-tuned"}

            combined_desc = " ".join(job.get('description', '').lower() for job in recent_roles)
            has_leadership = _any_kw(combined_desc, leadership_verbs)
            has_tech = _any_kw(combined_desc, tech_verbs)

            if has_leadership and not has_tech:
                return True, multiplier, "Senior role with no individual technical contribution evidence in the last 18 months"

        elif rule_id == "DQ4_title_inflation_hopper":
            # DQ4: Title-chasing by switching *companies* every ~1.5 years.
            # The original rule flagged any rapid promotion, including
            # well-deserved internal promotions at the same employer -- which
            # the JD does not object to. Only count it if the promotion
            # coincided with a company change.
            history = candidate.get('career_history', [])
            hierarchy = self.spec['disqualifier_rules']['DQ4_title_inflation_hopper']['title_hierarchy']

            promotions = 0
            for i in range(len(history) - 1):
                prior_job = history[i + 1]
                later_job = history[i]
                t1 = prior_job.get('title', '')
                t2 = later_job.get('title', '')
                same_company = prior_job.get('company', '') == later_job.get('company', '')
                if same_company:
                    continue  # internal promotion, not company-hopping for title

                try:
                    idx1 = next(idx for idx, v in enumerate(hierarchy) if v.lower() in t1.lower())
                    idx2 = next(idx for idx, v in enumerate(hierarchy) if v.lower() in t2.lower())
                    if idx2 > idx1 and later_job.get('duration_months', 0) < 18:
                        promotions += 1
                except StopIteration:
                    continue

            if promotions >= 2:
                return True, multiplier, "Title inflation detected: multiple rapid title jumps across company changes < 18mo"

        elif rule_id == "DQ5_pure_consulting_background":
            # DQ5: Pure consulting-firm career, no product-company experience.
            # Fixed two issues:
            #  1. Membership is checked against an explicit consulting-firm
            #     list (per the JD's named examples), not inferred from
            #     "absent from company_tiers.yaml" -- the latter would
            #     mis-flag any legitimate product company simply missing
            #     from the tiers file.
            #  2. Removed the "ownership keyword in description" escape --
            #     the JD's bar is *prior product-company employment*, not
            #     language like "shipped" appearing in a consulting-role
            #     description.
            history = candidate.get('career_history', [])
            if not history:
                return False, 1.0, ""

            product_cos = self._get_product_companies()

            def is_consulting(company: str) -> bool:
                c = company.lower()
                return any(firm in c for firm in self._consulting_firms)

            has_product_exp = any(job.get('company', '') in product_cos for job in history)
            all_known_consulting = all(is_consulting(job.get('company', '')) for job in history)

            if all_known_consulting and not has_product_exp:
                return True, multiplier, "Pure consulting background without product-company experience"

        elif rule_id == "DQ6_wrong_primary_domain":
            # DQ6: "People whose primary expertise is computer vision, speech,
            # or robotics without significant NLP/IR exposure."
            #
            # This was previously disabled outright, which let CV/speech/
            # robotics-only candidates rank as top fits for an NLP/IR/ranking
            # role. The fix here only penalizes candidates who have CV/speech/
            # robotics signal AND *zero* NLP/IR/ranking/search signal anywhere
            # in their career history -- this is the JD's actual bar
            # ("without significant NLP/IR exposure"), not "ever mentioned a
            # CV term," which is what caused false positives in an earlier
            # version of this logic.
            history = candidate.get('career_history', [])
            if not history:
                return False, 1.0, ""

            all_descs = " ".join(job.get('description', '').lower() for job in history)
            profile_summary = candidate.get('profile', {}).get('summary', '').lower()
            all_text = all_descs + " " + profile_summary

            has_cv_speech_robotics = _any_kw(all_text, self._cv_speech_robotics_kw)
            has_nlp_ir = _any_kw(all_text, self._nlp_ir_kw)

            if has_cv_speech_robotics and not has_nlp_ir:
                return True, multiplier, "Primary domain is computer vision/speech/robotics with no NLP/IR exposure"

        elif rule_id == "DQ7_closed_source_only":
            # DQ7: 5+ years, entirely closed-source, no external validation.
            # The JD's bar is "papers, talks, open-source" -- certifications
            # are not a substitute for that and have been removed as a signal.
            if candidate.get('profile', {}).get('years_of_experience', 0) < 5:
                return False, 1.0, ""

            signals = candidate.get('redrob_signals', {})
            github_score = signals.get('github_activity_score', -1)

            # jd_spec.json explicitly states:
            #   "redrob_signals.github_activity_score <= 10 OR == -1"
            # as the trigger condition for DQ7. -1 is the spec's sentinel
            # for "no GitHub data linked" and is intentionally treated the
            # same as a confirmed low score (0-10) — candidates with 5+ years
            # who haven't linked external validation should be flagged.
            # (Earlier code silently overrode this to 'skip on -1'; reversed.)
            has_known_low_github = (github_score == -1) or (0 <= github_score <= 10)

            all_text = (candidate.get('profile', {}).get('summary', '') + " " +
                        " ".join([j.get('description', '') for j in candidate.get('career_history', [])])).lower()
            oss_keywords = {"paper", "publication", "open source", "oss", "github", "conference", "blog", "talk", "speaker", "arxiv"}

            has_oss = _any_kw(all_text, oss_keywords)

            if has_known_low_github and not has_oss:
                return True, multiplier, "Low external validation for 5+ year experience (no papers/talks/open-source)"

        return False, 1.0, ""

    def _is_academic_employer(self, company: str) -> bool:
        if not company:
            return False
        company = company.lower()
        academic_keywords = {"university", "institute", "college", "laboratory", "national laboratory", "research center", "research centre"}
        # "lab" was previously matched as a raw substring, which fires on
        # "collaborate"/"collaboration" etc. Use the word-boundary-safe
        # "laboratory" form, plus an explicit "research lab" phrase.
        return _any_kw(company, academic_keywords) or _kw_in(company, "research lab")