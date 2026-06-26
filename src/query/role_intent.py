from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import json
from loguru import logger

# --- Basic Identity & Context ---

@dataclass
class LocationPolicy:
    primary: List[str]
    acceptable: List[str]
    relocation: str
    work_mode: str

@dataclass
class ExperienceRange:
    stated_range: List[int]
    strict: bool
    interpretation: str

@dataclass
class TeamContext:
    team_size_expected: str
    mentorship_required: bool
    scope: str

@dataclass
class RoleIdentity:
    role_title: str
    role_archetype: str
    company_stage: str
    company_type: str
    location: LocationPolicy
    employment_type: str
    experience_years: ExperienceRange
    team_context: TeamContext

# --- Competencies & Signals ---

@dataclass
class Signal:
    signal: str
    description: str
    weight: str
    evidence_patterns: List[str] = field(default_factory=list)
    tech_examples: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    models_mentioned_examples: List[str] = field(default_factory=list)

@dataclass
class Disqualifier:
    id: str
    description: str
    action: str
    evidence: List[str] = field(default_factory=list)
    firms: List[str] = field(default_factory=list)
    counter_signals: List[str] = field(default_factory=list)

@dataclass
class AntiPattern:
    id: str
    description: str
    pattern: str
    penalty: str
    evidence: List[str] = field(default_factory=list)

# --- Archetypes & Patterns ---

@dataclass
class IdealArchetype:
    total_experience_years: Dict[str, Any]
    applied_ml_ai_years: Dict[str, Any]
    company_archetype: str
    services_exposure: str
    key_achievement: str
    scale_expectation: str
    opinions_signal: str
    coding_recency: str
    leadership_signal: str
    tenure_intent: str
    location_flexibility: str

@dataclass
class CareerPattern:
    pattern_name: str
    description: str
    priority: str

# --- Policies & Weights ---

@dataclass
class NoticePolicy:
    ideal_max_days: int
    buyout_available: bool
    max_acceptable_days: int
    penalty_beyond_30_days: str

@dataclass
class BehavioralWeight:
    ideal: Optional[str] = None
    decay: Optional[str] = None
    penalty_below: Optional[str] = None
    penalty_above: Optional[str] = None
    boost: Optional[str] = None
    weight: Optional[str] = None

@dataclass
class ScoringPolicy:
    relevance_weight: float
    trust_coherence_weight: float
    availability_behavioral_weight: float
    career_archetype_weight: float
    location_availability_weight: float
    ranking_principle: str
    strictness: str

# --- Root Intent Object ---

@dataclass
class RoleIntent:
    role_identity: RoleIdentity
    mandate: List[Dict[str, Any]]
    required_signals: List[Signal]
    preferred_signals: List[Signal]
    hard_disqualifiers: List[Disqualifier]
    anti_patterns: List[AntiPattern]
    ideal_archetype: IdealArchetype
    career_patterns: List[CareerPattern]
    location_policy: LocationPolicy
    notice_policy: NoticePolicy
    behavioral_weights: Dict[str, BehavioralWeight]
    cultural_signals: Dict[str, str]
    scoring_policy: ScoringPolicy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class JDParser:
    """
    Converts a raw Job Description into a structured RoleIntent model.
    This is designed to be LLM-driven; the current implementation uses a
    simulated LLM output for the target role to maintain system functionality
    while providing the full target schema.
    """
    def __init__(self, jd_path: str):
        self.jd_path = jd_path
        self.intent: Optional[RoleIntent] = None

    def parse(self) -> RoleIntent:
        logger.info(f"Parsing JD from {self.jd_path} into robust intent model...")
        with open(self.jd_path, 'r') as f:
            text = f.read()

        # INTEGRATION POINT: In a production environment, this block would be replaced by:
        # response = llm_client.generate_structured_output(text, schema=RoleIntent)
        # return response

        # Simulated high-fidelity LLM extraction for "Senior AI Engineer — Founding Team"
        intent = RoleIntent(
            role_identity=RoleIdentity(
                role_title="Senior AI Engineer",
                role_archetype="founding_engineer_ai_ranking",
                company_stage="Series A",
                company_type="AI-native talent intelligence platform",
                location=LocationPolicy(
                    primary=["Pune", "Noida"],
                    acceptable=["Hyderabad", "Mumbai", "Delhi NCR", "Bangalore"],
                    relocation="preferred_from_tier_1_indian_cities",
                    work_mode="hybrid_flexible"
                ),
                employment_type="full_time",
                experience_years=ExperienceRange(
                    stated_range=[5, 9],
                    strict=False,
                    interpretation="proxy_for_judgment_not_tenure"
                ),
                team_context=TeamContext(
                    team_size_expected="4_to_12_growth",
                    mentorship_required=True,
                    scope="founding_member_of_ai_engineering_org"
                )
            ),
            mandate=[
                {"priority": "P0", "task": "Own intelligence layer (ranking, retrieval, matching)"},
                {"priority": "P0", "task": "Ship v2 ranking system (embeddings, hybrid retrieval)"},
                {"priority": "P1", "task": "Set up evaluation infrastructure (benchmarks, A/B tests)"},
                {"priority": "P1", "task": "Drive long-term matching architecture"},
                {"priority": "P2", "task": "Mentor next round of hires"},
                {"priority": "P2", "task": "Work with Recruiter-experience PM"},
            ],
            required_signals=[
                Signal(
                    signal="production_embeddings_retrieval",
                    description="Shipped embeddings-based retrieval to real users",
                    weight="critical",
                    evidence_patterns=["built", "shipped", "production", "deployed", "embedding drift", "index refresh", "retrieval quality regression"],
                    models_mentioned_examples=["sentence-transformers", "BGE", "E5", "OpenAI embeddings"]
                ),
                Signal(
                    signal="vector_or_hybrid_search_infrastructure",
                    description="Operational experience with vector DB or hybrid search",
                    weight="critical",
                    tech_examples=["Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", "FAISS"]
                ),
                Signal(
                    signal="ranking_evaluation_expertise",
                    description="Designed evaluation frameworks for ranking systems",
                    weight="critical",
                    metrics=["NDCG", "MRR", "MAP", "offline-to-online correlation", "A/B test interpretation"]
                ),
                Signal(
                    signal="strong_python_code_quality",
                    description="Strong Python with attention to code quality",
                    weight="high"
                )
            ],
            preferred_signals=[
                Signal(signal="llm_finetuning", description="LLM fine-tuning experience", weight="medium", examples=["LoRA", "QLoRA", "PEFT"]),
                Signal(signal="learning_to_rank", description="Learning-to-rank models", weight="medium", examples=["XGBoost ranker", "neural LTR"]),
                Signal(signal="hr_tech_or_marketplace", description="Prior exposure to HR-tech or marketplace", weight="low_to_medium"),
                Signal(signal="distributed_systems_or_inference_optimization", description="Distributed systems or inference optimization", weight="medium"),
                Signal(signal="open_source_contributions_ai_ml", description="Open-source contributions in AI/ML", weight="medium")
            ],
            hard_disqualifiers=[
                Disqualifier(id="pure_research_no_production", description="Academic/research-only environments without production deployment", action="reject_unless_strong_counter_evidence", evidence=["PhD in lab", "research scientist with no shipped products", "academic papers only"]),
                Disqualifier(id="langchain_only_recent", description="AI experience primarily under 12 months of LangChain/OpenAI wrapper projects", action="reject_unless_pre_llm_ml_production_experience", evidence=["LangChain", "recent OpenAI API projects", "tutorial-style demos"]),
                Disqualifier(id="no_recent_production_coding", description="Senior engineer who hasn't written production code in 18+ months", action="reject", evidence=["only architecture/tech lead", "no recent engineering output"]),
                Disqualifier(id="consulting_only_background", description="Entire career at consulting/services firms", action="reject_unless_prior_product_company_experience", firms=["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]),
                Disqualifier(id="wrong_domain_specialization", description="Primary expertise in CV/Speech/Robotics without NLP/IR", action="reject", evidence=["computer vision", "speech recognition", "robotics", "no NLP/IR/recsys"]),
                Disqualifier(id="closed_source_without_validation", description="5+ years closed-source proprietary systems without external validation", action="soft_reject", counter_signals=["papers", "talks", "open-source", "blog posts", "public portfolio"]),
            ],
            anti_patterns=[
                AntiPattern(id="title_chaser", description="Career optimized for title escalation via frequent job switching", pattern="Senior -> Staff -> Principal with avg tenure < 2 years", penalty="high"),
                AntiPattern(id="framework_enthusiast", description="Profile built around tutorials and hot frameworks", pattern="Framework-heavy blog posts, LangChain tutorials", penalty="medium_to_high", evidence=["LangChain tutorials", "demo projects"]),
                AntiPattern(id="skill_keyword_inflation", description="Long list of AI skills with no evidence of production systems", pattern="High skill count vs low evidence count", penalty="medium"),
            ],
            ideal_archetype=IdealArchetype(
                total_experience_years={"target": [6, 8], "acceptable": [5, 12]},
                applied_ml_ai_years={"target": [4, 5], "minimum": 3},
                company_archetype="product_companies",
                services_exposure="not_primary",
                key_achievement="shipped_end_to_end_ranking_search_or_recommendation_system_to_real_users",
                scale_expectation="meaningful_scale",
                opinions_signal="strong_defensible_opinions_on_retrieval_evaluation_llm_integration",
                coding_recency="active_production_coding_within_last_12_months",
                leadership_signal="mentored_or_led_small_teams_while_still_coding",
                tenure_intent="likely_to_stay_3_plus_years",
                location_flexibility="willing_to_relocate_to_pune_or_noida_or_already_there"
            ),
            career_patterns=[
                CareerPattern("Product-company DNA", "Consistent history of working at product-driven companies", "high"),
                CareerPattern("End-to-end ownership", "Evidence of owning a system from design to production", "high"),
                CareerPattern("Ranking/IR/recsys experience", "Direct experience with retrieval/ranking", "critical"),
                CareerPattern("Scaling experience", "Worked on systems with meaningful user scale", "medium"),
                CareerPattern("Stable progression", "Logical growth in responsibility and title", "medium"),
                CareerPattern("Coding seniority", "Active coding despite senior title", "high"),
            ],
            location_policy=LocationPolicy(
                primary=["Pune", "Noida"],
                acceptable=["Hyderabad", "Mumbai", "Delhi NCR", "Bangalore"],
                relocation="preferred_from_tier_1_indian_cities",
                work_mode="hybrid_flexible"
            ),
            notice_policy=NoticePolicy(
                ideal_max_days=30,
                buyout_available=True,
                max_acceptable_days=60,
                penalty_beyond_30_days="medium"
            ),
            behavioral_weights={
                "last_active_date": BehavioralWeight(ideal="within_30_days", decay="steep_after_90_days"),
                "recruiter_response_rate": BehavioralWeight(ideal=">0.20", penalty_below="high"),
                "avg_response_time_hours": BehavioralWeight(ideal="<72", penalty_above="high"),
                "open_to_work_flag": BehavioralWeight(boost="moderate"),
                "profile_views_received_30d": BehavioralWeight(weight="low"),
                "applications_submitted_30d": BehavioralWeight(weight="low_to_medium"),
                "interview_completion_rate": BehavioralWeight(ideal=">0.50", weight="medium"),
                "saved_by_recruiters_30d": BehavioralWeight(weight="low_to_medium"),
                "notice_period_days": BehavioralWeight(ideal="<30", penalty_above="medium"),
            },
            cultural_signals={
                "async_writing": "prefers_candidates_with_evidence_of_writing_or_documentation",
                "open_disagreement": "team_with_debate_culture",
                "fast_iteration": "comfortable_with_ambiguity_and_rapid_change",
                "not_big_tech_ladder": "rejects_candidates_seeking_meta_google_style_ladder",
                "scrappy_product_attitude": "values_shipper_over_researcher"
            },
            scoring_policy=ScoringPolicy(
                relevance_weight=0.40,
                trust_coherence_weight=0.25,
                availability_behavioral_weight=0.20,
                career_archetype_weight=0.10,
                location_availability_weight=0.05,
                ranking_principle="prefer_10_great_matches_over_1000_maybes",
                strictness="high_threshold_for_top_tier"
            )
        )

        self.intent = intent
        return intent
