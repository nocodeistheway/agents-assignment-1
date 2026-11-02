#!/usr/bin/env python3
"""
Verify RAG Setup

Runs a series of checks to ensure the RAG system is properly configured
and working correctly.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import chromadb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

console = Console()

COLLECTION_NAME = "agentic_ai_papers"


def check_vectorstore() -> tuple[bool, str]:
    """Check if vector store exists."""
    if not settings.vectorstore_path.exists():
        return False, f"Not found at {settings.vectorstore_path}"

    # Check for ChromaDB files
    chroma_files = list(settings.vectorstore_path.glob("*"))
    if not chroma_files:
        return False, "Directory exists but is empty"

    return True, f"Found at {settings.vectorstore_path}"


def check_collection() -> tuple[bool, str, dict]:
    """Check if collection is valid and get stats."""
    try:
        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()

        if count == 0:
            return False, "Collection is empty", {}

        # Get sample metadata to verify structure
        sample = collection.peek(1)

        return True, f"Collection '{COLLECTION_NAME}' loaded", {
            "total_chunks": count,
            "sample_metadata": sample.get("metadatas", [{}])[0] if sample.get("metadatas") else {}
        }
    except Exception as e:
        return False, f"Error loading collection: {e}", {}


def check_paper_coverage(collection_stats: dict) -> tuple[bool, str, dict]:
    """Check if all papers are indexed."""
    try:
        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        collection = client.get_collection(name=COLLECTION_NAME)

        # Get all unique paper IDs
        all_data = collection.get(include=["metadatas"])
        paper_ids = set()
        for meta in all_data.get("metadatas", []):
            if meta and "paper_id" in meta:
                paper_ids.add(meta["paper_id"])

        # Load paper index to compare
        import json
        index_path = settings.papers_path / "paper_index.json"
        with open(index_path) as f:
            index = json.load(f)

        expected_papers = {p["id"] for p in index["papers"]}
        indexed_papers = paper_ids

        missing = expected_papers - indexed_papers
        extra = indexed_papers - expected_papers

        if missing:
            return False, f"Missing papers: {', '.join(missing)}", {
                "indexed": len(indexed_papers),
                "expected": len(expected_papers),
                "missing": list(missing)
            }

        return True, f"{len(indexed_papers)}/{len(expected_papers)} papers indexed", {
            "indexed": len(indexed_papers),
            "expected": len(expected_papers)
        }
    except Exception as e:
        return False, f"Error checking coverage: {e}", {}


def check_embeddings() -> tuple[bool, str]:
    """Check if embedding provider is working."""
    try:
        from tools.embeddings import get_embedding_provider

        provider = get_embedding_provider()
        test_embedding = provider.embed_query("test")

        if len(test_embedding) != provider.dimension:
            return False, f"Unexpected dimension: {len(test_embedding)} vs {provider.dimension}"

        return True, f"{settings.embedding_provider.upper()} embeddings working"
    except Exception as e:
        return False, f"Error: {e}"


def run_test_query() -> tuple[bool, str, list]:
    """Run a test query and return results."""
    try:
        from tools.embeddings import get_embedding_provider

        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        collection = client.get_collection(name=COLLECTION_NAME)

        provider = get_embedding_provider()

        query = "What is an intelligent agent?"
        query_embedding = provider.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas", "distances"]
        )

        if not results.get("metadatas") or not results["metadatas"][0]:
            return False, "No results returned", []

        top_results = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            similarity = max(0, 1 - dist / 2)
            top_results.append({
                "paper_id": meta.get("paper_id", "unknown"),
                "section": meta.get("section", "unknown"),
                "score": similarity
            })

        return True, "Query successful", top_results
    except Exception as e:
        return False, f"Query error: {e}", []


def main():
    """Run all verification checks."""
    console.print(Panel(
        "[bold blue]RAG Setup Verification[/bold blue]\n\n"
        "Running checks to verify your setup is correct.",
        title="Research Crew"
    ))

    all_passed = True
    checks = []

    # Check 1: Vector store exists
    console.print("\n[1/5] Checking vector store...")
    passed, message = check_vectorstore()
    checks.append(("Vector store", passed, message))
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"      {status} {message}")
    if not passed:
        all_passed = False

    # Check 2: Collection loads
    console.print("\n[2/5] Loading collection...")
    passed, message, stats = check_collection()
    checks.append(("Collection", passed, message))
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"      {status} {message}")
    if stats.get("total_chunks"):
        console.print(f"      {stats['total_chunks']} total chunks")
    if not passed:
        all_passed = False

    # Check 3: Paper coverage
    console.print("\n[3/5] Verifying paper coverage...")
    passed, message, coverage = check_paper_coverage(stats)
    checks.append(("Paper coverage", passed, message))
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"      {status} {message}")
    if not passed:
        all_passed = False

    # Check 4: Embeddings working
    console.print("\n[4/5] Testing embedding provider...")
    passed, message = check_embeddings()
    checks.append(("Embeddings", passed, message))
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"      {status} {message}")
    if not passed:
        all_passed = False

    # Check 5: Test query
    console.print("\n[5/5] Running test query...")
    passed, message, results = run_test_query()
    checks.append(("Test query", passed, message))
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"      {status} {message}")

    if results:
        console.print(f"      Query: \"What is an intelligent agent?\"")
        console.print()

        table = Table(title="Top 3 Results")
        table.add_column("Paper", style="cyan")
        table.add_column("Section")
        table.add_column("Score", justify="right")

        for r in results:
            table.add_row(
                r["paper_id"],
                r["section"],
                f"{r['score']:.3f}"
            )

        console.print(table)

    if not passed:
        all_passed = False

    # Final summary
    console.print("\n")
    if all_passed:
        console.print(Panel(
            "[bold green]All Checks Passed![/bold green]\n\n"
            "Your RAG setup is ready for the assignment.\n"
            "Run [cyan]python main.py --example[/cyan] to test the crew.",
            title="Success"
        ))
    else:
        failed_checks = [c[0] for c in checks if not c[1]]
        console.print(Panel(
            f"[bold red]Some Checks Failed[/bold red]\n\n"
            f"Failed: {', '.join(failed_checks)}\n\n"
            "Please fix the issues and run verification again.",
            title="Verification Failed"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
