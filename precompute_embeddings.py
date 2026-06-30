import json
import hashlib
import numpy as np
import os
import sys
from loguru import logger
from src.retrieval.semantic import SemanticScorer
from sentence_transformers import SentenceTransformer
import argparse


def _md5_file(path: str) -> str:
    """Computes MD5 of a file in 8 KB streaming chunks."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--input", type=str, default="candidates.jsonl", help="Input candidates JSONL file")
    parser.add_argument("--out-npy", type=str, default="candidate_embeddings.npy", help="Output numpy file")
    parser.add_argument("--out-map", type=str, default="candidate_idx_map.json", help="Output index map JSON file")
    parser.add_argument("--out-hash", type=str, default="candidate_embeddings.hash", help="Output hash sidecar file")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if hash matches")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"Computing MD5 of {args.input}...")
    input_hash = _md5_file(args.input)

    # ── 1. Skip-if-unchanged check ───────────────────────────────────────────
    if not args.force and os.path.exists(args.out_hash):
        with open(args.out_hash, 'r') as f:
            stored_hash = f.read().strip()
        if stored_hash == input_hash and os.path.exists(args.out_npy) and os.path.exists(args.out_map):
            logger.info("Embeddings are already up to date. Use --force to rebuild.")
            sys.exit(0)

    logger.info(f"Loading candidates from {args.input}...")
    candidates = []
    with open(args.input, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))

    # ── 2. Build candidate texts and idx_map safely ──────────────────────────
    logger.info("Generating candidate texts...")
    candidate_texts = []
    candidate_ids = []
    idx_map = {}
    duplicate_count = 0
    missing_count = 0

    for i, cand in enumerate(candidates):
        cid = cand.get("candidate_id")
        if not cid:
            missing_count += 1
            cid = f"UNKNOWN_{i}" # Generate a dummy ID so it maps to something
        elif cid in idx_map:
            duplicate_count += 1
            logger.warning(f"Duplicate candidate_id found: {cid} at index {i}. Overwriting previous entry in idx_map.")

        candidate_ids.append(cid)
        idx_map[cid] = i
        
        # Call static method without instantiating SemanticScorer
        candidate_texts.append(SemanticScorer.create_candidate_text(cand))

    if len(idx_map) != len(candidates):
        logger.error(f"Integrity warning: {len(candidates)} candidates loaded, but idx_map has {len(idx_map)} entries.")
        if missing_count > 0:
            logger.error(f"  -> {missing_count} candidates were missing a candidate_id.")
        if duplicate_count > 0:
            logger.error(f"  -> {duplicate_count} candidate_ids were duplicates.")
        logger.error("Proceeding, but downstream semantic lookup for these candidates may be compromised.")

    # ── 3. Encode ─────────────────────────────────────────────────────────────
    logger.info("Encoding candidates using SentenceTransformer... this may take a while!")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(candidate_texts, show_progress_bar=True, convert_to_numpy=True)

    logger.info("Normalizing vectors...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    # ── 4. Atomic Writes ──────────────────────────────────────────────────────
    tmp_npy = args.out_npy + ".tmp"
    tmp_map = args.out_map + ".tmp"
    tmp_hash = args.out_hash + ".tmp"

    logger.info("Writing temporary files...")
    try:
        np.save(tmp_npy, norm_embeddings)
        with open(tmp_map, 'w') as f:
            json.dump(idx_map, f)
        with open(tmp_hash, 'w') as f:
            f.write(input_hash)
    except Exception as e:
        logger.error(f"Failed to write temporary files: {e}")
        for tmp_file in [tmp_npy, tmp_map, tmp_hash]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        sys.exit(1)

    logger.info("Swapping temporary files into place atomically...")
    os.replace(tmp_npy, args.out_npy)
    os.replace(tmp_map, args.out_map)
    os.replace(tmp_hash, args.out_hash)

    logger.info("Precomputation complete!")


if __name__ == "__main__":
    main()
