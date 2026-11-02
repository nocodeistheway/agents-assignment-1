"""
Evaluation Suite for RAG System

Provides metrics, evaluation runner, and report generation
to measure retrieval quality.
"""

from evals.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    concept_coverage,
    hit_rate,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "concept_coverage",
    "hit_rate",
]
