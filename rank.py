import argparse
import json
import csv
from loguru import logger
from src.pipeline import RankingPipeline
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

    # 4. Synthesize Reasoning
    for res in results:
        res['reasoning'] = synthesizer.synthesize(res)

    # 5. Audit Results
    if not auditor.audit(results):
        logger.error("Submission audit failed! Check honeypots.")
        return

    auditor.generate_manifest("submission_metadata.json")

    # 6. Write to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "final_score", "reasoning"])
        for res in results[:100]: # Top 100 only for submission
            writer.writerow([res['candidate_id'], f"{res['final_score']:.4f}", res['reasoning']])

    logger.info(f"Successfully ranked candidates. Results saved to {args.output}")

if __name__ == "__main__":
    main()
