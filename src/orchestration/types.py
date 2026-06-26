from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Evidence:
    text: str
    source: str  # e.g., "experience[0].description" or "summary"

@dataclass
class Verdict:
    agent: str
    signal: str  # "strong", "moderate", "weak", "none"
    confidence: float  # 0.0 to 1.0
    evidence: List[Evidence] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    reasoning: str = ""
    score: float = 0.0  # Normalized score for fallback/fusion

@dataclass
class CandidateEvaluation:
    candidate_id: str
    verdicts: Dict[str, Verdict] = field(default_factory=dict)
    trust_score: float = 1.0
    calculation: str = ""
    availability_score: float = 1.0
    risk_score: float = 0.0
    final_score: float = 0.0
    tier: str = "unlikely_fit"
    justification: str = ""
    key_risks: List[str] = field(default_factory=list)
