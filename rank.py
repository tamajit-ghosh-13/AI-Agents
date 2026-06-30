import argparse
import json
import csv
from loguru import logger
from src.orchestration.pipeline import RankingPipeline
from src.synthesis import ReasonSynthesizer
from src.auditor import SubmissionAuditor

def update_reasoning_prefix(tier: str, original_reasoning: str) -> str:
    import re
    
    # Map tier names to the required bracket labels
    tier_map = {
        "perfect_fit": "[perfect_fit]",
        "ideal_fit": "[ideal_fit]",
        "strong_fit": "[strong_fit]",
        "good_fit": "[good_fit]",
        "potential_fit": "[potential_fit]",
        "marginal_fit": "[marginal_fit]",
        "unlikely_fit": "[unlikely_fit]",
        "rejected": "[unlikely_fit]"
    }
    
    prefix = tier_map.get(tier, "[unlikely_fit]")
    return re.sub(r'^\[.*?\]', prefix, original_reasoning)

def load_candidates(file_path: str):
    candidates = []
    with open(file_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Redrob Intelligence Ranker v2.0")
    parser.add_argument("--input", type=str, default="candidates.jsonl", help="Input candidates file")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV file")
    parser.add_argument("--jd", type=str, default="job_description.txt", help="Full JD text file")
    args = parser.parse_args()

    logger.info("Initializing Redrob Intelligence Ranker v2.0...")

    # 1. Load Configuration & Data
    try:
        with open(args.jd, 'r') as f:
            jd_text = f.read()
        candidates = load_candidates(args.input)
    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        return

    # 2. Initialize Pipeline
    pipeline = RankingPipeline(
        spec_path="config/jd_spec.json",
        tiers_path="config/company_tiers.yaml"
    )
    synthesizer = ReasonSynthesizer()
    auditor = SubmissionAuditor()

    # 3. Execute Ranking
    results = pipeline.run(candidates, jd_text)

    # 4. Synthesize Reasoning for Top 100
    # Inject rank position so synthesizer can calibrate tone to position, not just tier.
    for i, res in enumerate(results[:100]):
        res['_rank'] = i + 1
        res['reasoning'] = synthesizer.synthesize(res)

    # Ensure we have reasoning for the audit if it needs it,
    # but auditor usually checks for honeypots/integrity.
    # Since the output CSV only takes top 100, we only synthesize top 100.

    # 5. Audit Results
    if not auditor.audit(results):
        logger.error("Submission audit failed! Check honeypots.")
        return

    auditor.generate_manifest("submission_metadata.json")

    # results is already sorted score DESC, candidate_id ASC from pipeline.run().
    # No re-sort needed.

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, res in enumerate(results[:100]):
            cid = res['candidate_id']
            rank = i + 1
            f_score_str = f"{res['final_score']:.4f}"
            trust_score = res.get('trust_score', '')
            reasoning_base = res.get('reasoning', '')
            tier = res.get('tier', 'unlikely_fit')
            new_reas = update_reasoning_prefix(tier, reasoning_base)
            final_reasoning = f"trust_score = {trust_score} | {new_reas}"
            writer.writerow([cid, rank, f_score_str, final_reasoning])


    logger.info(f"Successfully ranked candidates. Results saved to {args.output}")

if __name__ == "__main__":
    main()
