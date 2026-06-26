import json
import csv
from typing import List, Dict, Any
from loguru import logger

from src.orchestration.pipeline import RankingPipeline

def load_candidates(file_path: str) -> List[Dict[str, Any]]:
    candidates = []
    with open(file_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates

def main():
    spec_path = './config/jd_spec.json'
    tiers_path = './config/company_tiers.yaml'
    candidates_path = './candidates.jsonl'
    output_path = 'submissions.csv'

    logger.info("Loading candidates...")
    candidates = load_candidates(candidates_path)

    # We'll use a dummy JD text because the current JDParser
    # simulates output based on the spec_path anyway.
    jd_text = "Senior AI Engineer Role - See spec for details"

    logger.info("Initializing RIO-X Pipeline...")
    pipeline = RankingPipeline(spec_path, tiers_path)

    logger.info("Running pipeline...")
    results = pipeline.run(candidates, jd_text)

    logger.info(f"Processing complete. Writing to {output_path}...")

    # We output candidate_id and final_score as the primary requirements
    # adding tier and trust_score for completeness.
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['candidate_id', 'final_score', 'tier', 'trust_score'])

        for res in results:
            # Note: the pipeline return is a list of dicts
            # where candidate_data is nested.
            writer.writerow([
                res['candidate_id'],
                f"{res['final_score']:.4f}",
                res['tier'],
                f"{res['trust_score']:.4f}" if 'trust_score' in res else "N/A"
            ])

    logger.info(f"Successfully saved {len(results)} rankings to {output_path}")

if __name__ == "__main__":
    main()
