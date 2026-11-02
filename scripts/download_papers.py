#!/usr/bin/env python3
"""
Download Research Papers

Downloads all 15 curated papers from paper_index.json to the data/papers directory.
Supports resuming interrupted downloads and verifying PDF integrity.
"""

import json
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

console = Console()

# HTTP headers to avoid blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/pdf,*/*",
}


def load_paper_index() -> dict:
    """Load the paper index JSON file."""
    index_path = settings.papers_path / "paper_index.json"

    if not index_path.exists():
        console.print(f"[red]Error: paper_index.json not found at {index_path}[/red]")
        sys.exit(1)

    with open(index_path) as f:
        return json.load(f)


def verify_pdf(file_path: Path) -> bool:
    """Verify that a file is a valid PDF."""
    if not file_path.exists():
        return False

    # Check file size (should be at least 10KB for a real PDF)
    if file_path.stat().st_size < 10_000:
        return False

    # Check PDF magic bytes
    with open(file_path, "rb") as f:
        header = f.read(8)
        return header.startswith(b"%PDF")


def download_paper(paper: dict, output_dir: Path, force: bool = False) -> tuple[bool, str]:
    """
    Download a single paper.

    Args:
        paper: Paper metadata from index
        output_dir: Directory to save PDF
        force: Force re-download even if file exists

    Returns:
        Tuple of (success, message)
    """
    filename = paper["filename"]
    url = paper["download_url"]
    output_path = output_dir / filename

    # Check if already downloaded
    if output_path.exists() and not force:
        if verify_pdf(output_path):
            return True, "Already downloaded"
        else:
            console.print(f"[yellow]  Invalid PDF, re-downloading...[/yellow]")

    # Download the file
    try:
        response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower() and "pdf" not in content_type.lower():
            return False, "URL returned HTML instead of PDF"

        # Save the file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify the downloaded file
        if verify_pdf(output_path):
            file_size = output_path.stat().st_size / 1024
            return True, f"Downloaded ({file_size:.0f} KB)"
        else:
            output_path.unlink()  # Remove invalid file
            return False, "Downloaded file is not a valid PDF"

    except requests.exceptions.Timeout:
        return False, "Download timed out"
    except requests.exceptions.RequestException as e:
        return False, f"Download failed: {str(e)[:50]}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def main(force: bool = False):
    """Main download function."""
    console.print(Panel(
        "[bold blue]Downloading Research Papers[/bold blue]\n\n"
        "This will download all 15 curated papers on agentic AI.",
        title="Research Crew Setup"
    ))

    # Load paper index
    index = load_paper_index()
    papers = index["papers"]

    # Ensure output directory exists
    settings.papers_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\nSource: paper_index.json")
    console.print(f"Output: {settings.papers_path}")
    console.print(f"Papers: {len(papers)}\n")

    # Download each paper
    results = []
    total_size = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Downloading papers...", total=len(papers))

        for i, paper in enumerate(papers, 1):
            progress.update(task, description=f"[{i}/{len(papers)}] {paper['id']}")

            success, message = download_paper(paper, settings.papers_path, force)

            if success:
                file_path = settings.papers_path / paper["filename"]
                if file_path.exists():
                    total_size += file_path.stat().st_size / 1024

            results.append({
                "id": paper["id"],
                "filename": paper["filename"],
                "success": success,
                "message": message
            })

            progress.advance(task)

            # Small delay between downloads to be nice to servers
            if i < len(papers):
                time.sleep(0.5)

    # Display results
    console.print("\n")

    # Success table
    success_count = sum(1 for r in results if r["success"])
    failed = [r for r in results if not r["success"]]

    table = Table(title="Download Summary")
    table.add_column("Status", style="bold")
    table.add_column("Count")
    table.add_column("Details")

    table.add_row(
        "[green]Success[/green]",
        str(success_count),
        f"{total_size/1024:.1f} MB total"
    )

    if failed:
        table.add_row(
            "[red]Failed[/red]",
            str(len(failed)),
            ", ".join(r["id"] for r in failed)
        )

    console.print(table)

    # Show failed downloads
    if failed:
        console.print("\n[yellow]Failed Downloads:[/yellow]")
        for r in failed:
            console.print(f"  - {r['id']}: {r['message']}")
            paper = next(p for p in papers if p["id"] == r["id"])
            console.print(f"    URL: {paper['download_url']}")

        console.print("\n[dim]You may need to download these manually.[/dim]")

    # Final status
    if success_count == len(papers):
        console.print("\n[bold green]All papers downloaded successfully![/bold green]")
        console.print("\nNext step: Run [cyan]python scripts/setup_vectorstore.py[/cyan]")
    else:
        console.print(f"\n[yellow]Downloaded {success_count}/{len(papers)} papers.[/yellow]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download research papers")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download of all papers"
    )

    args = parser.parse_args()
    main(force=args.force)
