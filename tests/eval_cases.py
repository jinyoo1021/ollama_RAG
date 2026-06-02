"""Compatibility wrapper for retrieval evaluation cases."""

from src.evaluation import BLOCKCHAIN_EVAL_CASES

EVAL_CASES = [
    {
        "id": case.id,
        "query": case.query,
        "expected_pages": list(case.expected_refs[0].pages),
        "expected_terms": list(case.expected_terms),
    }
    for case in BLOCKCHAIN_EVAL_CASES
]
