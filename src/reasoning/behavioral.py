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

import math
from typing import Dict, Any
from datetime import datetime
from src.orchestration.types import Verdict, Evidence
from src.query.role_intent import RoleIntent

class BehavioralEvaluator:
    """
    Evaluates candidate availability and engagement based on recruiter signals.
    Utilizes policies directly from RoleIntent.
    """
    def __init__(self, intent: RoleIntent):
        self.intent = intent
        self.notice_policy = intent.notice_policy
        self.behavioral_weights = intent.behavioral_weights

    def evaluate(self, candidate: Dict[str, Any]) -> Verdict:
        """
        Computes the availability verdict based on notice period, responsiveness, and activity recency.
        """
        signals = candidate.get('redrob_signals', {})
        reasoning_parts = []
        risks = []
        evidence_list = []
        
        score_multiplier = 1.0

        # 1. Notice Period Penalty
        notice_days = signals.get('notice_period_days')
        if notice_days is not None:
            evidence_list.append(Evidence(text=f"Notice period: {notice_days} days", source="redrob_signals"))
            if notice_days > self.notice_policy.max_acceptable_days:
                score_multiplier *= 0.1  # Heavy penalty for being way outside acceptable range
                risks.append(f"Notice period ({notice_days}d) exceeds max acceptable ({self.notice_policy.max_acceptable_days}d).")
                reasoning_parts.append(f"Severely penalized for {notice_days}d notice period.")
            elif notice_days > self.notice_policy.ideal_max_days:
                score_multiplier *= 0.6  # Medium penalty
                risks.append(f"Notice period ({notice_days}d) is longer than ideal ({self.notice_policy.ideal_max_days}d).")
                reasoning_parts.append(f"Penalized for {notice_days}d notice period.")
            else:
                reasoning_parts.append(f"Good notice period ({notice_days}d).")

        # 2. Open to Work Boost
        open_to_work = signals.get('open_to_work_flag', False)
        if open_to_work:
            score_multiplier *= 1.2
            reasoning_parts.append("Boosted for 'Open to Work' flag.")

        # 3. Activity Recency
        last_active = signals.get('last_active_date', '')
        if not last_active:
            score_multiplier *= 0.5
            reasoning_parts.append("Penalty due to missing activity date.")
            risks.append("No recent activity data.")
        else:
            try:
                last_date = datetime.strptime(last_active, '%Y-%m-%d')
                days_since = max(0, (datetime.now() - last_date).days)
                
                # Use decay steepness from RoleIntent if available (e.g. steep_after_90_days)
                # For simplicity, we implement a standard decay curve
                if days_since <= 30:
                    activity_mult = 1.0
                elif days_since <= 90:
                    activity_mult = 0.9 - ((days_since - 30) * (0.4 / 60)) # decays to 0.5 at 90 days
                else:
                    activity_mult = 0.5
                    
                score_multiplier *= activity_mult
                reasoning_parts.append(f"Activity recency: {days_since} days ago.")
                if activity_mult < 0.6:
                    risks.append("Low engagement signal: candidate hasn't been active recently.")
            except ValueError:
                score_multiplier *= 0.5
                reasoning_parts.append("Invalid activity date format. Defaulting to high penalty.")

        # 4. Responsiveness
        response_rate = signals.get('recruiter_response_rate', 0.5)
        evidence_list.append(Evidence(text=f"Response rate: {response_rate:.2f}", source="redrob_signals"))
        
        # Base availability score derived from response rate
        base_score = 0.4 + (response_rate * 0.6)
        
        final_score = base_score * score_multiplier
        normalized_score = max(0.0, min(1.0, final_score))

        if normalized_score > 0.8: signal = "strong"
        elif normalized_score > 0.5: signal = "moderate"
        elif normalized_score > 0.2: signal = "weak"
        else: signal = "none"

        return Verdict(
            agent="AvailabilityAgent",
            signal=signal,
            confidence=1.0 if last_active else 0.5,
            evidence=evidence_list,
            risks=risks,
            reasoning=" ".join(reasoning_parts) + f" Final normalized availability score: {normalized_score:.2f}",
            score=normalized_score
        )
