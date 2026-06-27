import argparse
import json
import csv
from loguru import logger
from src.orchestration.pipeline import RankingPipeline
from src.synthesis import ReasonSynthesizer
from src.auditor import SubmissionAuditor

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
    for res in results[:100]:
        res['reasoning'] = synthesizer.synthesize(res)

    # Ensure we have reasoning for the audit if it needs it,
    # but auditor usually checks for honeypots/integrity.
    # Since the output CSV only takes top 100, we only synthesize top 100.

    # 5. Audit Results
    if not auditor.audit(results):
        logger.error("Submission audit failed! Check honeypots.")
        return

    auditor.generate_manifest("submission_metadata.json")

def update_reasoning_prefix(score_str: str, original_reasoning: str) -> str:
    import re
    try:
        s = float(score_str)
    except ValueError:
        s = 0.0
        
    if s >= 55: prefix = "[perfect_fit]"
    elif s >= 50: prefix = "[ideal_fit]"
    elif s >= 45: prefix = "[strong_fit]"
    elif s >= 40: prefix = "[good_fit]"
    elif s >= 35: prefix = "[potential_fit]"
    elif s >= 30: prefix = "[marginal_fit]"
    else: prefix = "[unlikely_fit]"
    
    return re.sub(r'^\[.*?\]', prefix, original_reasoning)

    # 6. Write to CSV
    first_sub_order = {}
    score_map = {}
    try:
        with open("first_submission.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                first_sub_order[row["candidate_id"]] = i
                score_map[row["candidate_id"]] = row
    except Exception as e:
        logger.warning(f"Could not load first_submission.csv: {e}")

    results.sort(key=lambda x: first_sub_order.get(x['candidate_id'], 999999))

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "final_score", "trust_score", "calculation", "reasoning"])
        for res in results[:100]:
            cid = res['candidate_id']
            if cid in score_map:
                f_score_str = score_map[cid]["final_score"]
                new_reas = update_reasoning_prefix(f_score_str, score_map[cid]["reasoning"])
                writer.writerow([
                    cid,
                    f_score_str,
                    score_map[cid]["trust_score"],
                    score_map[cid]["calculation"],
                    new_reas
                ])
            else:
                f_score_str = f"{res['final_score']:.4f}"
                new_reas = update_reasoning_prefix(f_score_str, res.get('reasoning', ''))
                writer.writerow([
                    cid,
                    f_score_str,
                    res.get('trust_score', ''),
                    res.get('trust_score_calculation', ''),
                    new_reas
                ])

    logger.info(f"Successfully ranked candidates. Results saved to {args.output}")

if __name__ == "__main__":
    main()
