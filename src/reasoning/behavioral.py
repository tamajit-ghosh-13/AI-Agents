from typing import Dict, Any
from datetime import datetime

class BehavioralEvaluator:
    """
    Evaluates candidate availability and engagement based on recruiter signals.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        # Configuration for decay thresholds and penalties
        self.config = config or {
            'decay_short': 30,
            'decay_medium': 90,
            'penalty_long': 0.5,
            'penalty_medium_start': 0.9,
            'penalty_medium_slope': 0.2 / 60,
            'resp_rate_base': 0.7,
            'resp_rate_range': 0.6
        }

from typing import Dict, Any
from datetime import datetime
from src.orchestration.types import Verdict, Evidence

class BehavioralEvaluator:
    """
    Evaluates candidate availability and engagement based on recruiter signals.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        # Configuration for decay thresholds and penalties
        self.config = config or {
            'decay_short': 30,
            'decay_medium': 90,
            'penalty_long': 0.5,
            'penalty_medium_start': 0.9,
            'penalty_medium_slope': 0.2 / 60,
            'resp_rate_base': 0.7,
            'resp_rate_range': 0.6
        }

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Computes the availability verdict based on responsiveness and activity recency.
        """
        signals = candidate.get('redrob_signals', {})

        # 1. Responsiveness
        response_rate = signals.get('recruiter_response_rate', 0.5)
        response_rate = max(0.0, min(1.0, response_rate))
        resp_mult = self.config['resp_rate_base'] + (response_rate * self.config['resp_rate_range'])

        # 2. Activity Decay
        last_active = signals.get('last_active_date', '')
        if not last_active:
            final_score = float(resp_mult * 0.5)
            reasoning = "High penalty applied due to missing activity date."
            activity_mult = 0.5
        else:
            try:
                last_date = datetime.strptime(last_active, '%Y-%m-%d')
                days_since = (datetime.now() - last_date).days
                days_since = max(0, days_since)

                if days_since <= self.config['decay_short']:
                    activity_mult = 1.0
                elif days_since <= self.config['decay_medium']:
                    activity_mult = self.config['penalty_medium_start'] - (days_since - self.config['decay_short']) * self.config['penalty_medium_slope']
                else:
                    activity_mult = self.config['penalty_long']

                final_score = float(resp_mult * activity_mult)
                reasoning = f"Activity recency: {days_since} days since last active. Multiplier: {activity_mult:.2f}."
            except ValueError:
                final_score = float(resp_mult * 0.5)
                reasoning = "Invalid activity date format. Defaulting to high penalty."
                activity_mult = 0.5

        normalized_score = min(1.0, final_score)

        if normalized_score > 0.8: signal = "strong"
        elif normalized_score > 0.5: signal = "moderate"
        elif normalized_score > 0.2: signal = "weak"
        else: signal = "none"

        risks = []
        if activity_mult < 0.6:
            risks.append("Low engagement signal: candidate hasn't been active recently.")

        return Verdict(
            agent="AvailabilityAgent",
            signal=signal,
            confidence=1.0 if last_active else 0.5,
            evidence=[Evidence(text=f"Response rate: {response_rate:.2f}", source="redrob_signals")],
            risks=risks,
            reasoning=f"{reasoning} Final normalized availability score: {normalized_score:.2f}",
            score=normalized_score
        )
