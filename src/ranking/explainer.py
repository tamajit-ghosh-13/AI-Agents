from typing import List, Dict, Any
from src.orchestration.types import CandidateEvaluation, Verdict, Evidence

class RankExplainer:
    """
    Generates human-readable justifications and risk summaries for candidates
    based on their reasoning chain and trust metrics.
    """

    def __init__(self):
        pass

    def explain(self, eval_obj: CandidateEvaluation) -> Dict[str, Any]:
        """
        Produces a structured explanation of the candidate's rank.
        """
        justification = self._build_narrative(eval_obj)
        key_risks = self._extract_key_risks(eval_obj)

        return {
            "justification": justification,
            "key_risks": key_risks
        }

    def _build_narrative(self, eval_obj: CandidateEvaluation) -> str:
        """
        Constructs a one-paragraph "Why ranked here" narrative.
        """
        # Start with a general sentiment based on the tier
        tier_messages = {
            "strong_fit": "Strongly recommended due to an exceptional alignment of technical depth and career trajectory.",
            "possible_fit": "A viable candidate with core competencies, though some gaps or risks were noted.",
            "unlikely_fit": "Not recommended for this specific role due to significant misalignments or trust issues.",
            "rejected": "Rejected based on hard-gate disqualifiers."
        }

        intro = tier_messages.get(eval_obj.tier, "Candidate evaluation completed.")

        # Gather strong signals
        strong_signals = [
            v.agent for agent, v in eval_obj.verdicts.items()
            if v.signal == "strong"
        ]

        signal_text = ""
        if strong_signals:
            signal_text = f" Notable strengths were found in {', '.join(strong_signals)}."

        # Incorporate trust
        trust_text = ""
        if eval_obj.trust_score < 0.7:
            trust_text = f" However, the profile shows coherence anomalies (Trust: {eval_obj.trust_score:.2f})."
        elif eval_obj.trust_score > 0.95:
            trust_text = " The profile is highly consistent and trustworthy."

        return f"{intro}{signal_text}{trust_text}"

    def _extract_key_risks(self, eval_obj: CandidateEvaluation) -> List[str]:
        """
        Aggregates risks from verdicts and coherence checks.
        """
        risks = []

        # 1. Risks from reasoning agents
        for verdict in eval_obj.verdicts.values():
            risks.extend(verdict.risks)

        # 2. Risks from the evaluation object itself (e.g. trust anomalies)
        risks.extend(eval_obj.key_risks)

        # Remove duplicates and limit to top 5
        unique_risks = list(dict.fromkeys(risks))
        return unique_risks[:5]
