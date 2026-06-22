import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import json

class SemanticScorer:
    def __init__(self, spec_path: str):
        # Load dynamic weights from the JD spec
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)

        self.required_skills = self.spec.get('required_skills', {})
        self.preferred_skills = self.spec.get('preferred_skills', {})

        # Lazy loading of model to avoid expensive init during every instantiation
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def create_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """
        Synthesizes candidate data into a single string for semantic matching.
        Gives more weight to recent roles.
        """
        profile = candidate.get('profile', {})
        history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])

        text_parts = [
            profile.get('headline', ''),
            profile.get('summary', ''),
        ]

        # Weighted history: Most recent roles are added first and repeated slightly
        for i, job in enumerate(history):
            title = job.get('title', '')
            desc = job.get('description', '')
            weight = 2 if i == 0 else (1 if i < 3 else 0.5)
            for _ in range(int(weight)):
                text_parts.append(f"{title} {desc}")

        for skill in skills:
            text_parts.append(f"{skill.get('name')} ({skill.get('proficiency')})")

        return " ".join(filter(None, text_parts))

    def score_candidates(self, candidates: List[Dict[str, Any]], jd_text: str) -> np.ndarray:
        """
        Computes semantic similarity between candidates and the full JD text.
        """
        if not candidates:
            return np.array([])

        # Use full JD text for embedding to avoid keyword-stuffing traps
        target_embedding = self.model.encode([jd_text])[0]

        # Batch encode candidates for speed
        candidate_texts = [self.create_candidate_text(c) for c in candidates]
        candidate_embeddings = self.model.encode(candidate_texts, show_progress_bar=False)

        # Normalize vectors for cosine similarity
        cand_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        cand_norms[cand_norms == 0] = 1.0
        norm_cand_embeddings = candidate_embeddings / cand_norms

        target_norm = np.linalg.norm(target_embedding)
        if target_norm == 0: return np.zeros(len(candidates))
        norm_target = target_embedding / target_norm

        # Dot product of normalized vectors is Cosine Similarity
        scores = np.dot(norm_cand_embeddings, norm_target)

        return scores

    def get_skill_match_score(self, candidate: Dict[str, Any]) -> float:
        """
        Hard-skill overlap score based on the spec.
        Normalizes by candidate skill count to penalize bloat.
        """
        skills_list = candidate.get('skills', [])
        if not skills_list:
            return 0.0

        cand_skills = {s.get('name', '').lower(): s.get('proficiency') for s in skills_list}
        score = 0.0
        total_weight = sum(self.required_skills.values()) + sum(self.preferred_skills.values())

        for skill, weight in self.required_skills.items():
            if skill.lower() in cand_skills:
                multiplier = {"expert": 1.2, "advanced": 1.0, "intermediate": 0.7, "beginner": 0.4}.get(cand_skills[skill.lower()], 0.5)
                score += weight * multiplier

        for skill, weight in self.preferred_skills.items():
            if skill.lower() in cand_skills:
                score += weight * 0.5

        # Normalize by candidate's total skill count to penalize "everything-is-a-skill" profiles
        bloat_penalty = 1.0 / (1.0 + max(0, len(skills_list) - 20) * 0.01)

        return (score / total_weight if total_weight > 0 else 0.0) * bloat_penalty
