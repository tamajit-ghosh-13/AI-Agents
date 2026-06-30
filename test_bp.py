import json
from loguru import logger
from typing import List, Dict, Any, Set

def detect_boilerplate(candidates: List[Dict[str, Any]]) -> Set[str]:
    ngram_to_cids = {}
    flagged: Set[str] = set()
    
    for cand in candidates:
        cid = cand.get('candidate_id')
        if not cid:
            continue
            
        for job in cand.get('career_history', []):
            desc = job.get('description', '').lower().strip()
            if not desc:
                continue
            
            tokens = desc.split()
            if len(tokens) < 10:
                continue
                
            # Generate 10-grams
            for i in range(len(tokens) - 9):
                # use tuple for memory efficiency
                ngram = tuple(tokens[i:i+10])
                if ngram not in ngram_to_cids:
                    ngram_to_cids[ngram] = set()
                ngram_to_cids[ngram].add(cid)

    for ngram, cids in ngram_to_cids.items():
        if len(cids) > 1:
            flagged.update(cids)
            
    return flagged

with open('candidate_sample.jsonl') as f:
    cands = [json.loads(line) for line in f]

f = detect_boilerplate(cands)
print("Flagged:", len(f))
