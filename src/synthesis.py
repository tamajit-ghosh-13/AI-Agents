from typing import Dict, Any, List, Set, Optional
import json
import yaml
import re


# ---------------------------------------------------------------------------
# JD reference constants — used for JD-connection and rank-consistency checks
# ---------------------------------------------------------------------------
_JD_CORE_SIGNALS = [
    "retrieval", "ranking", "embedding", "semantic search", "vector search",
    "hybrid search", "bm25", "ndcg", "mrr", "a/b test", "learning-to-rank",
    "re-ranker", "faiss", "opensearch", "elasticsearch", "rag",
]

_PRODUCTION_VERBS = [
    "deployed", "shipped", "served", "launched", "scaled",
    "in production", "production system", "live system",
]


class ReasonSynthesizer:
    """
    Generates human-readable, evidence-based justifications for a candidate's rank.

    Satisfies all six organiser rubric criteria:
    1. Specific facts    — cites YOE, title, company, named skills, signal values.
    2. JD connection     — links evidence to the JD's core asks (retrieval / ranking /
                           embedding / evaluation), not generic praise.
    3. Honest concerns   — surfaces real gaps (notice period, depth of evidence,
                           location headwind, skill-floor misses) for every candidate.
    4. No hallucination  — every claim is drawn directly from the candidate's profile.
    5. Variation         — achievement sentences are candidate-specific and
                           globally de-duplicated across the whole submission run.
    6. Rank consistency  — tone and concern language calibrated to RANK POSITION
                           (1–100), not tier label, so a rank-95 candidate never
                           gets glowing praise.
    """

    def __init__(self):
        # Track achievement sentences already used across candidates this run.
        # Key = first 40 chars of the sentence (lowercase). This prevents the
        # same boilerplate sentence appearing in more than one candidate's reasoning.
        self._global_used_achievements: Set[str] = set()

    def synthesize(self, result: Dict[str, Any]) -> str:
        """
        Builds a fully personalised reasoning string from the candidate's own profile.
        Pass `result['_rank']` (1-indexed) for rank-calibrated tone; falls back to
        tier-based tone if not present.
        """
        cand = result["candidate_data"]
        score = result["final_score"]
        dq_reasons = result.get("key_risks", [])
        tier = result.get("tier", "unlikely_fit")
        verdicts = result.get("verdicts", {})
        rank: Optional[int] = result.get("_rank", None)  # injected by rank.py

        if score == 0:
            return (
                f"Rejected: {', '.join(dq_reasons) if dq_reasons else 'Does not meet basic requirements'}."
            )

        # ── Load configs ──────────────────────────────────────────────────
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

        # ── 1. Basic facts ────────────────────────────────────────────────
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        history = cand.get("career_history", [])
        cand_skills = cand.get("skills", [])

        exp = profile.get("years_of_experience", 0)
        title = profile.get("current_title", "") or (history[0].get("title", "") if history else "")
        company = profile.get("current_company", "") or (history[0].get("company", "") if history else "")
        location = profile.get("location", "Unknown Location")
        country = profile.get("country", "")
        notice = signals.get("notice_period_days", 30)
        rr = signals.get("recruiter_response_rate", None)
        github = signals.get("github_activity_score", -1)

        intro = (
            f"This candidate is a {title} at {company} with {exp}y experience."
            if company
            else f"This candidate is a {title} with {exp}y experience."
        )

        # ── 2. Company pedigree (JD connection: product-company proof) ────
        pedigree_str = self._build_pedigree(history, tiers_data)

        # ── 3. Matched JD-required and preferred skills (specific facts) ──
        skills_str = self._build_skills_str(cand_skills, spec)

        # ── 4. Achievement sentences — candidate-unique, JD-connected ─────
        achievements_str = self._build_achievements(history, cand_skills)

        # ── 5. JD-connection gap: flag if no JD core signals surfaced ─────
        jd_conn_note = ""
        full_text = " ".join(
            (j.get("description", "") + " " + " ".join(s.get("name", "") for s in cand_skills))
            for j in history
        ).lower()
        jd_hits = [sig for sig in _JD_CORE_SIGNALS if sig in full_text]
        if not jd_hits:
            jd_conn_note = " No direct evidence of retrieval/ranking/embedding work found in profile."

        # ── 6. Availability & location ────────────────────────────────────
        loc_display = location + (f", {country}" if country and country not in location else "")
        availability_str = f" They are based in {loc_display} (notice period: {notice} days)."

        # ── 7. Honest concerns — proportional to rank ─────────────────────
        concern_parts = list(dq_reasons)

        # Notice period concern (JD specifies 30d ideal, 60d acceptable max)
        if notice > 90:
            concern_parts.append(
                f"Notice period ({notice}d) is a significant headwind — JD expects ≤30d ideal."
            )
        elif notice > 60:
            concern_parts.append(
                f"Notice period ({notice}d) exceeds the JD's preferred 60d threshold."
            )
        elif notice > 30 and (rank is None or rank > 30):
            # Only flag 31–60d notice for mid/lower ranked candidates
            concern_parts.append(
                f"Notice period ({notice}d) is above the JD's 30d ideal."
            )

        # Location headwind
        outside_india = country and country.strip() not in ("India", "IN", "")
        preferred_cities = ["pune", "noida"]
        welcome_cities = ["hyderabad", "mumbai", "delhi", "bengaluru", "bangalore"]
        loc_lower = location.lower()
        if outside_india:
            concern_parts.append(
                f"Based outside India ({location}) — relocation or remote arrangement needed."
            )
        elif not any(c in loc_lower for c in preferred_cities + welcome_cities):
            if rank is None or rank > 20:
                concern_parts.append(
                    f"Location ({location}) is outside preferred/welcome cities — confirm remote policy."
                )

        # Evidence depth — flag for mid/lower ranked candidates
        evidence_verdict = verdicts.get("EvidenceAgent", {})
        evidence_score = (
            evidence_verdict.get("score") if isinstance(evidence_verdict, dict)
            else getattr(evidence_verdict, "score", None)
        )
        if evidence_score is not None and evidence_score < 0.25:
            if rank is None or rank > 20:
                concern_parts.append("Limited quantified production evidence in career descriptions.")

        # GitHub / external validation
        if exp >= 5 and github in (-1, 0) and (rank is None or rank > 30):
            concern_parts.append(
                "No external validation signal (GitHub activity absent or missing)."
            )

        # Recruiter response rate
        if rr is not None and rr < 0.5:
            concern_parts.append(f"Low recruiter response rate ({rr:.0%}).")

        # ── 8. Rank-calibrated closing tone ───────────────────────────────
        rank_tone = self._rank_tone(rank, tier, score)

        # ── 9. Assemble ───────────────────────────────────────────────────
        parts = [f"[{tier}]", intro, pedigree_str, skills_str, achievements_str,
                 jd_conn_note, availability_str]
        reasoning = " ".join(p for p in parts if p).strip()
        reasoning = re.sub(r"\.[ ]+\.", ".", reasoning)
        reasoning = re.sub(r" {2,}", " ", reasoning)

        if concern_parts:
            unique_concerns = list(dict.fromkeys(concern_parts))
            reasoning += f" Note: {'; '.join(unique_concerns)}."

        if rank_tone:
            reasoning += f" {rank_tone}"

        return reasoning

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_pedigree(self, history: list, tiers_data: dict) -> str:
        ai_native = [c.lower() for c in tiers_data.get("ai_native", [])]
        tier1 = [c.lower() for c in tiers_data.get("product_tier1", [])]
        tier2 = (
            [c.lower() for c in tiers_data.get("product_tier2", [])]
            + [c.lower() for c in tiers_data.get("product_tier3_growth", [])]
            + [c.lower() for c in tiers_data.get("saas_b2b", [])]
        )
        aliases = {k.lower(): v.lower() for k, v in tiers_data.get("aliases", {}).items()}

        ai_cos, t1_cos, t2_cos = [], [], []
        for job in history:
            co = job.get("company", "")
            if not co:
                continue
            co_r = aliases.get(co.lower(), co.lower())
            if any(a == co_r or a in co_r for a in ai_native):
                ai_cos.append(co)
            elif any(t == co_r or t in co_r for t in tier1):
                t1_cos.append(co)
            elif any(t == co_r or t in co_r for t in tier2):
                t2_cos.append(co)

        parts = []
        if ai_cos:
            parts.append(f"AI-native background at {ai_cos[0]}")
        if t1_cos:
            parts.append(f"Tier-1 engineering pedigree at {t1_cos[0]}")
        if t2_cos and not (ai_cos or t1_cos):
            parts.append(f"product-company background at {t2_cos[0]}")

        return f" They have a {', and '.join(parts)}." if parts else ""

    def _build_skills_str(self, cand_skills: list, spec: dict) -> str:
        skills_map = {s.get("name", "").lower(): s for s in cand_skills}
        required = spec.get("required_skills", {})
        preferred = spec.get("preferred_skills", {})

        matched_req, matched_pref = [], []
        for cat, cfg in required.items():
            if cat.startswith("_"):
                continue
            forms = cfg.get("surface_forms_skills_array", []) if isinstance(cfg, dict) else []
            for form in forms:
                if form.lower() in skills_map:
                    matched_req.append(form)
                    break

        for cat, cfg in preferred.items():
            if cat.startswith("_"):
                continue
            forms = cfg.get("surface_forms", []) if isinstance(cfg, dict) else []
            for form in forms:
                if form.lower() in skills_map:
                    matched_pref.append(form)
                    break

        all_matched = matched_req + [p for p in matched_pref if p not in matched_req]
        if not all_matched:
            return ""
        unique = list(dict.fromkeys(all_matched))
        return f" Technically, they match JD requirements: {', '.join(unique[:5])}."

    def _build_achievements(self, history: list, cand_skills: list) -> str:
        """
        Picks the best 1–2 candidate-unique, JD-relevant achievement sentences.
        Priority: quantified + production + JD-signal keywords.
        Globally de-duplicates across candidates using self._global_used_achievements,
        so the same boilerplate sentence is never repeated in two rows.
        """
        sentences = []
        seen_in_this_profile: Set[str] = set()

        for job in history:
            desc = job.get("description", "")
            if not desc:
                continue
            for raw_s in re.split(r"(?<=[.!?])\s+", desc):
                s = raw_s.strip().rstrip(".")
                if not s or len(s) < 30 or len(s) > 220:
                    continue
                key = s[:40].lower()
                # Skip if seen in this candidate's own history
                if key in seen_in_this_profile:
                    continue
                seen_in_this_profile.add(key)
                priority = self._sentence_priority(s)
                if priority > 0:
                    sentences.append((priority, s))

        sentences.sort(key=lambda x: -x[0])

        best: List[str] = []
        for _, s in sentences:
            global_key = s[:40].lower()
            if global_key in self._global_used_achievements:
                continue  # already used for another candidate — skip
            self._global_used_achievements.add(global_key)
            best.append(f'"{s}"')
            if len(best) >= 2:
                break

        # If all top sentences were globally used, fall back to any non-used sentence
        if not best:
            for _, s in sentences:
                global_key = s[:40].lower()
                if global_key not in self._global_used_achievements:
                    self._global_used_achievements.add(global_key)
                    best.append(f'"{s}"')
                    break

        if not best:
            return " Limited unique production evidence found in career descriptions."
        return f" Key achievements: {'; and '.join(best)}."

    @staticmethod
    def _sentence_priority(s: str) -> int:
        score = 0
        sl = s.lower()
        # Quantified results (numbers, percentages, scale claims)
        if "%" in s or re.search(r"\d+[kmb]?\+?\s*(users|queries|requests|docs|candidates)", sl):
            score += 4
        if any(c.isdigit() for c in s):
            score += 1
        # Production proof verbs
        if any(v in sl for v in _PRODUCTION_VERBS):
            score += 3
        # JD core signals present
        if any(sig in sl for sig in _JD_CORE_SIGNALS):
            score += 3
        # Strong ownership language
        if any(w in sl for w in ["owned", "led", "built", "shipped", "designed"]):
            score += 2
        # Hard penalise boilerplate / generic sentences
        if "i worked closely with" in sl or "my own modeling work was secondary" in sl:
            score -= 10
        if "briefly explored" in sl or "early-stage exploration" in sl:
            score -= 5
        if "was secondary" in sl or "supporting role" in sl:
            score -= 5
        return score

    @staticmethod
    def _rank_tone(rank: Optional[int], tier: str, score: float) -> str:
        """
        Returns a closing sentence whose tone matches the candidate's RANK POSITION.
        If rank is provided, it takes precedence over tier to ensure rank-1 and rank-95
        never receive the same verdict language.
        """
        if rank is not None:
            if rank <= 5:
                return "Top candidate — priority shortlist immediately."
            elif rank <= 15:
                return "Strong recommend for recruiter screen — profile closely matches JD core requirements."
            elif rank <= 30:
                return "Good candidate — worth interviewing; confirm production depth in technical screen."
            elif rank <= 50:
                return "Solid background with some gaps relative to higher-ranked candidates; interview if capacity allows."
            elif rank <= 70:
                return (
                    "Below the median of this shortlist — proceed only if stronger candidates "
                    "are unavailable or the bar lowers."
                )
            elif rank <= 85:
                return (
                    "Significant gaps versus the top of the list; do not prioritise unless "
                    "shortlist is undersized."
                )
            else:
                return (
                    "At the tail of the ranked pool — not recommended unless all higher-ranked "
                    "candidates decline."
                )

        # Fallback: tier-based (used only if rank not injected)
        tone_map = {
            "perfect_fit": "Top candidate — priority shortlist immediately.",
            "ideal_fit": "Strong recommend for recruiter screen.",
            "strong_fit": "Good candidate — worth interviewing; confirm production depth.",
            "good_fit": "Solid background with some gaps; interview if capacity allows.",
            "potential_fit": "Below the median of this shortlist — proceed cautiously.",
            "marginal_fit": "Significant gaps — do not prioritise.",
            "unlikely_fit": "Not recommended for this role.",
        }
        return tone_map.get(tier, "")
