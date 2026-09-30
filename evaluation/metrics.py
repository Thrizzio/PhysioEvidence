import math
from typing import List, Sequence

def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Computes Recall@K: proportion of relevant documents found in top-k results.
    """
    if not relevant:
        return 0.0
    top_k_retrieved = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(top_k_retrieved.intersection(relevant_set))
    return hits / len(relevant_set)

def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Computes Precision@K: proportion of top-k retrieved documents that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k_retrieved if item in relevant_set)
    return hits / k

def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for a single query.
    Returns 1 / rank of the first relevant document retrieved, or 0.0 if none found.
    """
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (nDCG@K) using binary relevance.
    """
    if not relevant or k <= 0:
        return 0.0

    relevant_set = set(relevant)
    top_k_retrieved = retrieved[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item in enumerate(top_k_retrieved):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed, denominator log2(rank+1)

    # Calculate Ideal DCG (IDCG)
    idcg = 0.0
    ideal_hits = min(len(relevant_set), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
