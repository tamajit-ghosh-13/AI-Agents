from typing import Dict, Any
from datetime import datetime

class BehavioralEvaluator:
    """
    Evaluates candidate availability and engagement based on recruiter signals.
    """
    def __init__(self, config: Dict[str, Any] = None):
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

    def calculate_availability_multiplier(self, signals: Dict[str, Any]) -> float:
        """
        Computes a multiplier based on responsiveness and activity recency.
        """
        # 1. Responsiveness
        # Bounds check: response_rate must be in [0, 1]
        response_rate = signals.get('recruiter_response_rate', 0.5)
        response_rate = max(0.0, min(1.0, response_rate))

        # Response rate multiplier: [0.7 to 1.3]
        resp_mult = self.config['resp_rate_base'] + (response_rate * self.config['resp_rate_range'])

        # 2. Activity Decay
        last_active = signals.get('last_active_date', '')
        if not last_active:
            return float(resp_mult * 0.5) # High penalty for no activity date

        try:
            last_date = datetime.strptime(last_active, '%Y-%m-%d')
            days_since = (datetime.now() - last_date).days
            # Handle future dates by clamping to 0
            days_since = max(0, days_since)
        except ValueError:
            days_since = 365

        # Non-linear decay
        if days_since <= self.config['decay_short']:
            activity_mult = 1.0
        elif days_since <= self.config['decay_medium']:
            activity_mult = self.config['penalty_medium_start'] - (days_since - self.config['decay_short']) * self.config['penalty_medium_slope']
        else:
            activity_mult = self.config['penalty_long']

        return float(resp_mult * activity_mult)

    def get_score(self, candidate: Dict[str, Any]) -> float:
        """
        Returns the behavioral multiplier for a candidate.
        """
        signals = candidate.get('redrob_signals', {})
        return self.calculate_availability_multiplier(signals)
