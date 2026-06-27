import json
import numpy as np
from loguru import logger
from src.retrieval.semantic import SemanticScorer
from sentence_transformers import SentenceTransformer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--input", type=str, default="candidates.jsonl", help="Input candidates JSONL file")
    parser.add_argument("--out-npy", type=str, default="candidate_embeddings.npy", help="Output numpy file")
    parser.add_argument("--out-map", type=str, default="candidate_idx_map.json", help="Output index map JSON file")
    args = parser.parse_args()

    logger.info(f"Loading candidates from {args.input}...")
    candidates = []
    with open(args.input, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))

    # Initialize a dummy semantic scorer just to use its create_candidate_text method
    scorer = SemanticScorer("config/jd_spec.json")

    logger.info("Generating candidate texts...")
    candidate_texts = []
    candidate_ids = []
    idx_map = {}
    for i, cand in enumerate(candidates):
        cid = cand.get("candidate_id")
        candidate_ids.append(cid)
        idx_map[cid] = i
        candidate_texts.append(scorer.create_candidate_text(cand))

    logger.info("Encoding candidates using SentenceTransformer... this may take a while!")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Use show_progress_bar=True to give the user visual feedback during the long run
    embeddings = model.encode(candidate_texts, show_progress_bar=True, convert_to_numpy=True)
    
    logger.info("Normalizing vectors...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    logger.info(f"Saving embeddings to {args.out_npy}...")
    np.save(args.out_npy, norm_embeddings)

    logger.info(f"Saving index map to {args.out_map}...")
    with open(args.out_map, 'w') as f:
        json.dump(idx_map, f)

    logger.info("Precomputation complete!")

if __name__ == "__main__":
    main()
