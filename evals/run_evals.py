#!/usr/bin/env python3
"""
RAG Evaluation Runner

Runs the full evaluation suite against the vector store
and generates a detailed report.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

import chromadb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from tools.embeddings import get_embedding_provider
from evals.retrieval_metrics import EvaluationResult, aggregate_results
from evals.report_generator import generate_report

console = Console()

COLLECTION_NAME = "agentic_ai_papers"


def load_test_queries() -> list[dict]:
    """Load test queries from JSON file."""
    queries_path = settings.evals_path / "test_queries.json"

    if not queries_path.exists():
        console.print(f"[red]Error: test_queries.json not found at {queries_path}[/red]")
        sys.exit(1)

    with open(queries_path) as f:
        data = json.load(f)
        return data.get("queries", [])


def run_single_query(
    collection,
    provider,
    query_data: dict,
    k: int = 5
) -> EvaluationResult:
    """Run a single evaluation query."""
    query = query_data["query"]
    query_id = query_data.get("id", "unknown")
    expected_papers = query_data.get("expected_papers", [])
    expected_concepts = query_data.get("expected_concepts", [])

    # Time the query
    start_time = time.time()

    # Generate embedding
    query_embedding = provider.embed_query(query)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    latency_ms = (time.time() - start_time) * 1000

    # Extract paper IDs from results
    retrieved_papers = []
    retrieved_text = ""

    if results.get("metadatas") and results["metadatas"][0]:
        for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
            paper_id = meta.get("paper_id", "unknown")
            if paper_id not in retrieved_papers:
                retrieved_papers.append(paper_id)
            retrieved_text += doc + "\n\n"

    return EvaluationResult(
        query_id=query_id,
        query=query,
        retrieved_papers=retrieved_papers,
        expected_papers=expected_papers,
        retrieved_text=retrieved_text,
        expected_concepts=expected_concepts,
        latency_ms=latency_ms
    )


def group_results_by_category(results: list[EvaluationResult], queries: list[dict]) -> dict:
    """Group results by query category."""
    # Create a mapping of query_id to category
    category_map = {q["id"]: q.get("category", "other") for q in queries}

    grouped = {}
    for result in results:
        category = category_map.get(result.query_id, "other")
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(result)

    return grouped


def main(k: int = 5, verbose: bool = False):
    """Run the full evaluation suite."""
    console.print(Panel(
        "[bold blue]Running RAG Evaluation Suite[/bold blue]",
        title="Research Crew Evaluation"
    ))

    # Load test queries
    queries = load_test_queries()
    console.print(f"\nQueries: {len(queries)} | k={k} | Provider: {settings.embedding_provider.upper()}\n")

    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        console.print(f"[red]Error loading vector store: {e}[/red]")
        console.print("Run 'python scripts/setup_vectorstore.py' first.")
        sys.exit(1)

    # Initialize embedding provider
    try:
        provider = get_embedding_provider()
    except Exception as e:
        console.print(f"[red]Error initializing embeddings: {e}[/red]")
        sys.exit(1)

    # Run evaluations
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating queries...", total=len(queries))

        for query_data in queries:
            progress.update(task, description=f"Query: {query_data['id']}")

            result = run_single_query(collection, provider, query_data, k)
            results.append(result)

            if verbose:
                status = "[green]HIT[/green]" if result.hit else "[red]MISS[/red]"
                console.print(f"  {query_data['id']}: {status} (R@5={result.recall_5:.2f})")

            progress.advance(task)

    # Aggregate results
    overall = aggregate_results(results)

    # Display overall metrics
    console.print("\n")
    console.print("[bold]OVERALL METRICS[/bold]")
    console.print("=" * 40)

    metrics_table = Table(show_header=False)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    metrics_table.add_row("Recall@5", f"{overall['recall_5']*100:.1f}%")
    metrics_table.add_row("Precision@5", f"{overall['precision_5']*100:.1f}%")
    metrics_table.add_row("Mean Reciprocal Rank", f"{overall['mrr']:.2f}")
    metrics_table.add_row("Concept Coverage", f"{overall['concept_coverage']*100:.1f}%")
    metrics_table.add_row("Hit Rate", f"{overall['hit_rate']*100:.1f}%")
    metrics_table.add_row("Avg Latency", f"{overall['avg_latency_ms']:.0f}ms")

    console.print(metrics_table)

    # Group by category
    grouped = group_results_by_category(results, queries)

    console.print("\n[bold]CATEGORY BREAKDOWN[/bold]")
    console.print("=" * 40)

    cat_table = Table()
    cat_table.add_column("Category")
    cat_table.add_column("Recall@5", justify="right")
    cat_table.add_column("MRR", justify="right")
    cat_table.add_column("Queries", justify="right")

    for category, cat_results in sorted(grouped.items()):
        cat_metrics = aggregate_results(cat_results)
        cat_table.add_row(
            category.replace("_", " ").title(),
            f"{cat_metrics['recall_5']*100:.1f}%",
            f"{cat_metrics['mrr']:.2f}",
            str(len(cat_results))
        )

    console.print(cat_table)

    # Show failed queries
    failed = [r for r in results if r.recall_5 < 1.0]
    if failed:
        console.print("\n[bold yellow]QUERIES WITH INCOMPLETE RECALL[/bold yellow]")
        console.print("=" * 40)

        for r in failed[:5]:  # Show first 5
            console.print(f"\nQuery: \"{r.query}\"")
            console.print(f"  Expected: {', '.join(r.expected_papers)}")
            console.print(f"  Retrieved: {', '.join(r.retrieved_papers[:3])}")
            console.print(f"  Recall@5: {r.recall_5:.2f}")

        if len(failed) > 5:
            console.print(f"\n... and {len(failed) - 5} more")

    # Generate and save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = settings.outputs_path / f"eval_report_{timestamp}.md"
    settings.outputs_path.mkdir(parents=True, exist_ok=True)

    report_content = generate_report(results, queries, overall, grouped)
    report_path.write_text(report_content)

    console.print(f"\n[dim]Detailed report saved to: {report_path}[/dim]")

    # Return success/failure based on metrics
    if overall["hit_rate"] >= 0.8:
        console.print("\n[bold green]Evaluation complete - metrics look good![/bold green]")
    else:
        console.print("\n[bold yellow]Evaluation complete - some metrics below target.[/bold yellow]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-query results"
    )

    args = parser.parse_args()
    main(k=args.top_k, verbose=args.verbose)
