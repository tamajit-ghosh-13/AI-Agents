from typing import Dict, Any
import numpy as np
from datetime import datetime

class BehavioralEvaluator:
    def __init__(self):
        pass

    def calculate_availability_multiplier(self, signals: Dict[str, Any]) -> float:
        
        # Uses a non-linear decay for activity and a multiplier for response rate.
        
        # 1. Responsiveness
        response_rate = signals.get('recruiter_response_rate', 0.5)
        # Response rate multiplier: [0.7 to 1.3]
        resp_mult = 0.7 + (response_rate * 0.6)

        # 2. Activity Decay
        last_active = signals.get('last_active_date', '')
        if not last_active:
            return resp_mult * 0.5 # High penalty for no activity date

        try:
            last_date = datetime.strptime(last_active, '%Y-%m-%d')
            days_since = (datetime.now() - last_date).days
        except ValueError:
            days_since = 365

        # Non-linear decay:
        # 0-30 days: 1.0
        # 30-90 days: 0.9 -> 0.7
        # 90+ days: 0.5
        if days_since <= 30:
            activity_mult = 1.0
        elif days_since <= 90:
            activity_mult = 0.9 - (days_since - 30) * (0.2 / 60)
        else:
            activity_mult = 0.5

        return float(resp_mult * activity_mult)

    def get_score(self, candidate: Dict[str, Any]) -> float:
        signals = candidate.get('redrob_signals', {})
        return self.calculate_availability_multiplier(signals)
