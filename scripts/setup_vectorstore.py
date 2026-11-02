#!/usr/bin/env python3
"""
Setup Vector Store

Processes all PDFs, extracts text, chunks documents, generates embeddings,
and stores everything in ChromaDB for RAG retrieval.
"""

import json
import sys
import time
import hashlib
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
from config.chunking import CHUNKING_CONFIG
from tools.pdf_processor import extract_text_from_pdf, get_pdf_info
from tools.chunker import chunk_sections, estimate_embedding_cost, Chunk
from tools.embeddings import get_embedding_provider

console = Console()

COLLECTION_NAME = "agentic_ai_papers"


def load_paper_index() -> dict:
    """Load the paper index."""
    index_path = settings.papers_path / "paper_index.json"
    with open(index_path) as f:
        return json.load(f)


def get_cache_path(paper_id: str) -> Path:
    """Get the cache file path for extracted text."""
    return settings.processed_path / f"{paper_id}_extracted.json"


def load_cached_sections(paper_id: str) -> list | None:
    """Load cached extracted sections if available."""
    cache_path = get_cache_path(paper_id)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_cached_sections(paper_id: str, sections: list):
    """Save extracted sections to cache."""
    cache_path = get_cache_path(paper_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    sections_data = [
        {
            "paper_id": s.paper_id,
            "section_name": s.section_name,
            "text": s.text,
            "page_numbers": s.page_numbers
        }
        for s in sections
    ]

    with open(cache_path, "w") as f:
        json.dump(sections_data, f, indent=2)


def process_papers(papers: list, force: bool = False) -> tuple[list[Chunk], dict]:
    """
    Process all papers: extract text and create chunks.

    Returns:
        Tuple of (all_chunks, stats)
    """
    all_chunks = []
    stats = {
        "papers_processed": 0,
        "total_pages": 0,
        "total_sections": 0,
        "total_tokens": 0,
        "from_cache": 0,
    }

    console.print("\n[bold]Step 1/4: Extracting text from PDFs...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing papers...", total=len(papers))

        for paper in papers:
            paper_id = paper["id"]
            filename = paper["filename"]
            title = paper["title"]
            pdf_path = settings.papers_path / filename

            progress.update(task, description=f"Processing {paper_id}...")

            # Check if PDF exists
            if not pdf_path.exists():
                console.print(f"  [yellow]Skipping {paper_id}: PDF not found[/yellow]")
                progress.advance(task)
                continue

            # Try to load from cache
            cached = load_cached_sections(paper_id) if not force else None

            if cached:
                # Reconstruct section objects from cache
                from tools.pdf_processor import ExtractedSection
                sections = [
                    ExtractedSection(
                        paper_id=s["paper_id"],
                        section_name=s["section_name"],
                        text=s["text"],
                        page_numbers=s["page_numbers"]
                    )
                    for s in cached
                ]
                stats["from_cache"] += 1
            else:
                # Extract from PDF
                try:
                    sections = extract_text_from_pdf(pdf_path, paper_id)
                    save_cached_sections(paper_id, sections)
                except Exception as e:
                    console.print(f"  [red]Error processing {paper_id}: {e}[/red]")
                    progress.advance(task)
                    continue

            # Get page info
            try:
                pdf_info = get_pdf_info(pdf_path)
                stats["total_pages"] += pdf_info["page_count"]
            except:
                pass

            stats["total_sections"] += len(sections)

            # Create chunks
            paper_chunks = chunk_sections(sections, title, CHUNKING_CONFIG)
            all_chunks.extend(paper_chunks)

            stats["papers_processed"] += 1
            progress.advance(task)

    # Calculate total tokens
    for chunk in all_chunks:
        stats["total_tokens"] += chunk.token_count

    return all_chunks, stats


def create_vectorstore(chunks: list[Chunk]) -> dict:
    """
    Create ChromaDB vector store with embeddings.

    Returns:
        Statistics about the operation
    """
    console.print("\n[bold]Step 2/4: Chunking documents...[/bold]")
    console.print(f"  Created {len(chunks)} chunks")
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
    console.print(f"  Average chunk size: {avg_tokens:.0f} tokens\n")

    # Initialize embedding provider
    console.print("[bold]Step 3/4: Generating embeddings...[/bold]")
    provider = get_embedding_provider()
    console.print(f"  Provider: {settings.embedding_provider.upper()}")
    console.print(f"  Model: {provider.model_name}")

    # Estimate cost
    cost_info = estimate_embedding_cost(chunks, provider.model_name)
    if cost_info["estimated_cost_usd"] > 0:
        console.print(f"  Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")
    console.print()

    # Generate embeddings
    texts = [chunk.to_embedding_text() for chunk in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(texts))

        # Process in batches for progress updates
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = provider.embed_documents(batch)
            all_embeddings.extend(embeddings)
            progress.update(task, completed=min(i + batch_size, len(texts)))

    console.print()
    console.print("[bold]Step 4/4: Building ChromaDB index...[/bold]")

    # Initialize ChromaDB
    settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.vectorstore_path))

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for insertion
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # Create unique ID
        chunk_id = f"{chunk.paper_id}_{chunk.chunk_index}_{hashlib.md5(chunk.text[:100].encode()).hexdigest()[:8]}"
        ids.append(chunk_id)
        documents.append(chunk.to_embedding_text())
        metadatas.append({
            "paper_id": chunk.paper_id,
            "paper_title": chunk.paper_title,
            "section": chunk.section,
            "page_numbers": ",".join(map(str, chunk.page_numbers)),
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        })

    # Add to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )

    console.print(f"  Saved to {settings.vectorstore_path}")

    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": len(chunks),
        "embedding_dimension": provider.dimension,
    }


def main(force: bool = False):
    """Main setup function."""
    start_time = time.time()

    console.print(Panel(
        "[bold blue]Building Vector Store for Agentic AI Papers[/bold blue]\n\n"
        f"Embedding Provider: {settings.embedding_provider.upper()}",
        title="Research Crew Setup"
    ))

    # Validate settings
    errors = settings.validate()
    if errors:
        console.print("\n[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print("\n[dim]Check your .env file and try again.[/dim]")
        sys.exit(1)

    # Load paper index
    index = load_paper_index()
    papers = index["papers"]

    console.print(f"\nPapers to process: {len(papers)}")

    # Process papers
    chunks, extraction_stats = process_papers(papers, force)

    if not chunks:
        console.print("\n[red]No chunks created. Make sure PDFs are downloaded.[/red]")
        console.print("Run: [cyan]python scripts/download_papers.py[/cyan]")
        sys.exit(1)

    # Create vector store
    store_stats = create_vectorstore(chunks)

    # Calculate storage size
    storage_size = sum(
        f.stat().st_size for f in settings.vectorstore_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)  # MB

    elapsed = time.time() - start_time

    # Display summary
    console.print("\n")
    console.print(Panel(
        "[bold green]Setup Complete![/bold green]",
        title="Success"
    ))

    table = Table(title="Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Papers indexed", str(extraction_stats["papers_processed"]))
    table.add_row("Total pages", str(extraction_stats["total_pages"]))
    table.add_row("Total chunks", str(store_stats["total_chunks"]))
    table.add_row("Total tokens", f"{extraction_stats['total_tokens']:,}")
    table.add_row("Embedding dimension", str(store_stats["embedding_dimension"]))
    table.add_row("Storage size", f"{storage_size:.1f} MB")
    table.add_row("Time elapsed", f"{elapsed:.1f} seconds")
    if extraction_stats["from_cache"] > 0:
        table.add_row("From cache", str(extraction_stats["from_cache"]))

    console.print(table)

    console.print("\nNext step: Run [cyan]python scripts/verify_setup.py[/cyan] to test your setup.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup vector store for RAG")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-processing of all papers (ignore cache)"
    )

    args = parser.parse_args()
    main(force=args.force)
