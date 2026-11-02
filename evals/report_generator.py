"""
Evaluation Report Generator

Generates detailed markdown reports from evaluation results.
"""

from datetime import datetime
from evals.retrieval_metrics import EvaluationResult, aggregate_results


def generate_report(
    results: list[EvaluationResult],
    queries: list[dict],
    overall_metrics: dict,
    grouped_results: dict
) -> str:
    """
    Generate a detailed markdown evaluation report.

    Args:
        results: List of EvaluationResult objects
        queries: Original query definitions
        overall_metrics: Aggregated overall metrics
        grouped_results: Results grouped by category

    Returns:
        Markdown report as string
    """
    lines = []

    # Header
    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Queries:** {len(results)}")
    lines.append("")

    # Overall Metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Recall@5 | {overall_metrics['recall_5']*100:.1f}% |")
    lines.append(f"| Precision@5 | {overall_metrics['precision_5']*100:.1f}% |")
    lines.append(f"| Mean Reciprocal Rank | {overall_metrics['mrr']:.3f} |")
    lines.append(f"| Concept Coverage | {overall_metrics['concept_coverage']*100:.1f}% |")
    lines.append(f"| Hit Rate | {overall_metrics['hit_rate']*100:.1f}% |")
    lines.append(f"| Average Latency | {overall_metrics['avg_latency_ms']:.0f}ms |")
    lines.append("")

    # Category Breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Recall@5 | Precision@5 | MRR | Hit Rate | Queries |")
    lines.append("|----------|----------|-------------|-----|----------|---------|")

    for category, cat_results in sorted(grouped_results.items()):
        cat_metrics = aggregate_results(cat_results)
        lines.append(
            f"| {category.replace('_', ' ').title()} | "
            f"{cat_metrics['recall_5']*100:.1f}% | "
            f"{cat_metrics['precision_5']*100:.1f}% | "
            f"{cat_metrics['mrr']:.2f} | "
            f"{cat_metrics['hit_rate']*100:.1f}% | "
            f"{len(cat_results)} |"
        )

    lines.append("")

    # Detailed Results
    lines.append("## Detailed Results")
    lines.append("")

    # Perfect queries
    perfect = [r for r in results if r.recall_5 == 1.0]
    if perfect:
        lines.append("### Successful Queries (Recall@5 = 100%)")
        lines.append("")
        for r in perfect[:10]:
            lines.append(f"- **{r.query_id}**: \"{r.query}\"")
            lines.append(f"  - Retrieved: {', '.join(r.retrieved_papers[:3])}")
            lines.append(f"  - MRR: {r.mrr:.2f}")
        if len(perfect) > 10:
            lines.append(f"- ... and {len(perfect) - 10} more")
        lines.append("")

    # Failed queries
    failed = [r for r in results if r.recall_5 < 1.0]
    if failed:
        lines.append("### Queries Needing Improvement (Recall@5 < 100%)")
        lines.append("")
        for r in sorted(failed, key=lambda x: x.recall_5):
            lines.append(f"#### {r.query_id}")
            lines.append(f"- **Query:** \"{r.query}\"")
            lines.append(f"- **Expected:** {', '.join(r.expected_papers)}")
            lines.append(f"- **Retrieved:** {', '.join(r.retrieved_papers[:5])}")
            lines.append(f"- **Recall@5:** {r.recall_5:.2f}")
            lines.append(f"- **MRR:** {r.mrr:.2f}")

            # Identify what's missing
            expected_set = set(r.expected_papers)
            retrieved_set = set(r.retrieved_papers)
            missing = expected_set - retrieved_set
            if missing:
                lines.append(f"- **Missing:** {', '.join(missing)}")

            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Generate recommendations based on metrics
    recommendations = []

    if overall_metrics['recall_5'] < 0.8:
        recommendations.append(
            "- **Low Recall:** Consider increasing chunk overlap or trying different chunking strategies "
            "to ensure relevant content isn't split awkwardly."
        )

    if overall_metrics['mrr'] < 0.7:
        recommendations.append(
            "- **Low MRR:** Relevant results aren't appearing first. Consider adjusting embedding model "
            "or adding query expansion."
        )

    if overall_metrics['concept_coverage'] < 0.7:
        recommendations.append(
            "- **Low Concept Coverage:** Retrieved chunks may not contain expected terminology. "
            "Review chunking boundaries near key definitions."
        )

    # Category-specific recommendations
    for category, cat_results in grouped_results.items():
        cat_metrics = aggregate_results(cat_results)
        if cat_metrics['recall_5'] < 0.7:
            recommendations.append(
                f"- **{category.replace('_', ' ').title()} underperforming:** "
                f"Review the papers in this category and ensure key concepts are properly extracted."
            )

    if recommendations:
        for rec in recommendations:
            lines.append(rec)
    else:
        lines.append("- All metrics look healthy! Continue monitoring as corpus grows.")

    lines.append("")

    # Query distribution
    lines.append("## Query Distribution")
    lines.append("")
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")

    total = len(results)
    for category, cat_results in sorted(grouped_results.items()):
        pct = len(cat_results) / total * 100
        lines.append(f"| {category.replace('_', ' ').title()} | {len(cat_results)} | {pct:.1f}% |")

    lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Report generated by Research Crew Evaluation Suite*")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test report generation
    from evals.retrieval_metrics import EvaluationResult

    test_results = [
        EvaluationResult(
            query_id="test_1",
            query="What is an agent?",
            retrieved_papers=["paper_a", "paper_b"],
            expected_papers=["paper_a"],
            expected_concepts=["agent", "autonomy"],
            retrieved_text="An agent is autonomous...",
            latency_ms=100
        ),
        EvaluationResult(
            query_id="test_2",
            query="How does ReAct work?",
            retrieved_papers=["paper_c"],
            expected_papers=["paper_c", "paper_d"],
            expected_concepts=["reasoning", "acting"],
            retrieved_text="ReAct combines reasoning...",
            latency_ms=120
        ),
    ]

    test_queries = [
        {"id": "test_1", "category": "definitional"},
        {"id": "test_2", "category": "reasoning"},
    ]

    overall = aggregate_results(test_results)
    grouped = {
        "definitional": [test_results[0]],
        "reasoning": [test_results[1]],
    }

    report = generate_report(test_results, test_queries, overall, grouped)
    print(report)
