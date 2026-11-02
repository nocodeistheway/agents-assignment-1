"""
Retrieval Metrics for RAG Evaluation

Implements standard information retrieval metrics:
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Concept Coverage
- Hit Rate
"""

import re
from typing import Sequence


def recall_at_k(
    retrieved_papers: Sequence[str],
    expected_papers: Sequence[str],
    k: int | None = None
) -> float:
    """
    Calculate Recall@k: fraction of expected papers in top-k results.

    Args:
        retrieved_papers: List of retrieved paper IDs (in order)
        expected_papers: List of expected/relevant paper IDs
        k: Number of top results to consider (None = use all)

    Returns:
        Recall score between 0 and 1
    """
    if not expected_papers:
        return 1.0  # Nothing to recall

    if k is not None:
        retrieved_papers = retrieved_papers[:k]

    retrieved_set = set(retrieved_papers)
    expected_set = set(expected_papers)

    if not expected_set:
        return 1.0

    hits = len(retrieved_set & expected_set)
    return hits / len(expected_set)


def precision_at_k(
    retrieved_papers: Sequence[str],
    expected_papers: Sequence[str],
    k: int | None = None
) -> float:
    """
    Calculate Precision@k: fraction of top-k results that are relevant.

    Args:
        retrieved_papers: List of retrieved paper IDs (in order)
        expected_papers: List of expected/relevant paper IDs
        k: Number of top results to consider (None = use all)

    Returns:
        Precision score between 0 and 1
    """
    if k is not None:
        retrieved_papers = retrieved_papers[:k]

    if not retrieved_papers:
        return 0.0

    retrieved_set = set(retrieved_papers)
    expected_set = set(expected_papers)

    hits = len(retrieved_set & expected_set)
    return hits / len(retrieved_papers)


def mean_reciprocal_rank(
    retrieved_papers: Sequence[str],
    expected_papers: Sequence[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank: 1/rank of first relevant result.

    Args:
        retrieved_papers: List of retrieved paper IDs (in order)
        expected_papers: List of expected/relevant paper IDs

    Returns:
        MRR score between 0 and 1
    """
    if not expected_papers:
        return 1.0

    expected_set = set(expected_papers)

    for rank, paper_id in enumerate(retrieved_papers, 1):
        if paper_id in expected_set:
            return 1.0 / rank

    return 0.0


def hit_rate(
    retrieved_papers: Sequence[str],
    expected_papers: Sequence[str]
) -> bool:
    """
    Check if at least one expected paper appears in results.

    Args:
        retrieved_papers: List of retrieved paper IDs
        expected_papers: List of expected/relevant paper IDs

    Returns:
        True if there's at least one hit, False otherwise
    """
    if not expected_papers:
        return True

    retrieved_set = set(retrieved_papers)
    expected_set = set(expected_papers)

    return len(retrieved_set & expected_set) > 0


def concept_coverage(
    retrieved_text: str,
    expected_concepts: Sequence[str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate what fraction of expected concepts appear in retrieved text.

    Args:
        retrieved_text: Combined text from retrieved documents
        expected_concepts: List of concepts/terms expected to be found
        case_sensitive: Whether matching is case-sensitive

    Returns:
        Coverage score between 0 and 1
    """
    if not expected_concepts:
        return 1.0

    if not retrieved_text:
        return 0.0

    if not case_sensitive:
        retrieved_text = retrieved_text.lower()
        expected_concepts = [c.lower() for c in expected_concepts]

    found = 0
    for concept in expected_concepts:
        # Use word boundary matching for more accurate detection
        pattern = r'\b' + re.escape(concept) + r'\b'
        if re.search(pattern, retrieved_text, re.IGNORECASE if not case_sensitive else 0):
            found += 1

    return found / len(expected_concepts)


def ndcg_at_k(
    retrieved_papers: Sequence[str],
    expected_papers: Sequence[str],
    k: int | None = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if in expected_papers, 0 otherwise).

    Args:
        retrieved_papers: List of retrieved paper IDs (in order)
        expected_papers: List of expected/relevant paper IDs
        k: Number of top results to consider

    Returns:
        NDCG score between 0 and 1
    """
    import math

    if k is not None:
        retrieved_papers = retrieved_papers[:k]

    if not retrieved_papers or not expected_papers:
        return 0.0

    expected_set = set(expected_papers)

    # Calculate DCG
    dcg = 0.0
    for i, paper_id in enumerate(retrieved_papers, 1):
        rel = 1.0 if paper_id in expected_set else 0.0
        dcg += rel / math.log2(i + 1)

    # Calculate ideal DCG (all relevant docs first)
    ideal_order = sorted(retrieved_papers, key=lambda x: x in expected_set, reverse=True)
    idcg = 0.0
    for i, paper_id in enumerate(ideal_order, 1):
        rel = 1.0 if paper_id in expected_set else 0.0
        idcg += rel / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


class EvaluationResult:
    """Container for evaluation results of a single query."""

    def __init__(
        self,
        query_id: str,
        query: str,
        retrieved_papers: list[str],
        expected_papers: list[str],
        retrieved_text: str = "",
        expected_concepts: list[str] | None = None,
        latency_ms: float = 0.0
    ):
        self.query_id = query_id
        self.query = query
        self.retrieved_papers = retrieved_papers
        self.expected_papers = expected_papers
        self.retrieved_text = retrieved_text
        self.expected_concepts = expected_concepts or []
        self.latency_ms = latency_ms

        # Calculate all metrics
        self.recall_5 = recall_at_k(retrieved_papers, expected_papers, k=5)
        self.precision_5 = precision_at_k(retrieved_papers, expected_papers, k=5)
        self.mrr = mean_reciprocal_rank(retrieved_papers, expected_papers)
        self.hit = hit_rate(retrieved_papers, expected_papers)
        self.coverage = concept_coverage(retrieved_text, self.expected_concepts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "retrieved_papers": self.retrieved_papers,
            "expected_papers": self.expected_papers,
            "metrics": {
                "recall_5": self.recall_5,
                "precision_5": self.precision_5,
                "mrr": self.mrr,
                "hit": self.hit,
                "concept_coverage": self.coverage,
                "latency_ms": self.latency_ms,
            }
        }


def aggregate_results(results: list[EvaluationResult]) -> dict:
    """
    Aggregate multiple evaluation results into summary statistics.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary with aggregated metrics
    """
    if not results:
        return {}

    n = len(results)

    return {
        "num_queries": n,
        "recall_5": sum(r.recall_5 for r in results) / n,
        "precision_5": sum(r.precision_5 for r in results) / n,
        "mrr": sum(r.mrr for r in results) / n,
        "hit_rate": sum(1 for r in results if r.hit) / n,
        "concept_coverage": sum(r.coverage for r in results) / n,
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }


if __name__ == "__main__":
    # Quick test
    retrieved = ["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"]
    expected = ["paper_b", "paper_f"]

    print("Test metrics:")
    print(f"  Recall@5: {recall_at_k(retrieved, expected, k=5):.2f}")
    print(f"  Precision@5: {precision_at_k(retrieved, expected, k=5):.2f}")
    print(f"  MRR: {mean_reciprocal_rank(retrieved, expected):.2f}")
    print(f"  Hit Rate: {hit_rate(retrieved, expected)}")

    text = "An intelligent agent is capable of autonomy and reactivity."
    concepts = ["agent", "autonomy", "proactivity", "social ability"]
    print(f"  Concept Coverage: {concept_coverage(text, concepts):.2f}")
