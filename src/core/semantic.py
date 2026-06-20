import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from loguru import logger
import json

class SemanticScorer:
    def __init__(self, spec_path: str):
        # Load dynamic weights from the JD spec
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        # Use a tiny, fast, CPU-optimized model
        # all-MiniLM-L6-v2 is the gold standard for speed/performance trade-off
        logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.required_skills = self.spec.get('required_skills', {})
        self.preferred_skills = self.spec.get('preferred_skills', {})

    def create_candidate_text(self, candidate: Dict[str, Any]) -> str:
        # \"\"\"
        # Synthesizes candidate data into a a single string for semantic matching.
        # Focuses on summary, headline, and career descriptions.
        # \"\"\"
        profile = candidate.get('profile', {})
        history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])

        text_parts = [
            profile.get('headline', ''),
            profile.get('summary', ''),
        ]

        for job in history:
            text_parts.append(job.get('title', ''))
            text_parts.append(job.get('description', ''))

        for skill in skills:
            text_parts.append(f"{skill.get('name')} ({skill.get('proficiency')})")

        return " ".join(filter(None, text_parts))

    def score_candidates(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        # \"\"\"
        # Computes semantic similarity between candidates and the JD requirements.
        # \"\"\"
        if not candidates:
            return np.array([])

        # Create a 'target' string representing the ideal candidate
        target_text = " ".join(list(self.required_skills.keys()) + list(self.preferred_skills.keys()))
        target_embedding = self.model.encode([target_text])[0]

        # Batch encode candidates for speed
        candidate_texts = [self.create_candidate_text(c) for c in candidates]
        candidate_embeddings = self.model.encode(candidate_texts, show_progress_bar=False)

        # Use FAISS for lightning-fast cosine similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(candidate_embeddings)
        target_normalized = target_embedding / target_embedding.norm()
        
        # Dot product of normalized vectors is Cosine Similarity
        scores = np.dot(candidate_embeddings, target_normalized)

        return scores

    def get_skill_match_score(self, candidate: Dict[str, Any]) -> float:
        # \"\"\"
        # Hard-skill overlap score based on the spec.
        # \"\"\"
        cand_skills = {s.get('name').lower(): s.get('proficiency') for s in candidate.get('skills', [])}
        score = 0.0
        total_weight = sum(self.required_skills.values()) + sum(self.preferred_skills.values())

        for skill, weight in self.required_skills.items():
            if skill.lower() in cand_skills:
                # Proficiency multiplier
                multiplier = {"expert": 1.2, "advanced": 1.0, "intermediate": 0.7, "beginner": 0.4}.get(cand_skills[skill.lower()], 0.5)
                score += weight * multiplier

        for skill, weight in self.preferred_skills.items():
            if skill.lower() in cand_skills:
                score += weight * 0.5 # Preferred skills have less impact

        return score / total_weight if total_weight > 0 else 0.0
