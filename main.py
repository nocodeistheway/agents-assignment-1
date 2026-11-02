#!/usr/bin/env python3
"""
Research & Report Crew - Main Entry Point

Usage:
    python main.py "Your research question here"
    python main.py --interactive
    python main.py --example

Examples:
    python main.py "What are the current approaches to reducing hallucinations in LLMs?"
    python main.py "How do multi-agent systems coordinate and communicate?"

TODO: This file is mostly complete. You may customize it if needed,
but the main work is in agents/, tasks/, and crew.py.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables BEFORE importing crewai
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from crew import run_research


console = Console()

# Example research questions for students to try
EXAMPLE_QUESTIONS = [
    "What are the main approaches to building AI agents that can reason and act?",
    "How do multi-agent systems coordinate and communicate?",
    "What role does memory play in AI agent architectures?",
    "How does retrieval-augmented generation improve language model outputs?",
    "What are the key challenges in making AI agents safe and aligned?",
]


def save_report(report: str, question: str) -> Path:
    """Save the report to the outputs directory."""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a short filename from the question
    short_name = question[:30].replace(" ", "_").replace("?", "").replace("/", "_")
    filename = f"report_{short_name}_{timestamp}.md"

    filepath = outputs_dir / filename
    filepath.write_text(report)

    return filepath


def run_interactive():
    """Run in interactive mode, prompting for question."""
    console.print(Panel(
        "[bold blue]Research & Report Crew[/bold blue]\n\n"
        "This tool will search a corpus of 15 foundational AI agent papers\n"
        "and generate a literature review on your research question.",
        title="Welcome"
    ))

    console.print("\n[bold]Example questions:[/bold]")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        console.print(f"  {i}. {q}")

    console.print("\n")
    question = console.input("[bold green]Enter your research question:[/bold green] ")

    if not question.strip():
        console.print("[red]No question provided. Exiting.[/red]")
        sys.exit(1)

    return question.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Research & Report Crew - Generate literature reviews from AI agent papers"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Research question to investigate"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--example", "-e",
        action="store_true",
        help="Run with an example question"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the report to a file"
    )

    args = parser.parse_args()

    # Determine the question
    if args.interactive:
        question = run_interactive()
    elif args.example:
        question = EXAMPLE_QUESTIONS[0]
        console.print(f"[dim]Using example question: {question}[/dim]\n")
    elif args.question:
        question = args.question
    else:
        # No question provided, run interactive
        question = run_interactive()

    # Display header
    console.print("\n")
    console.print(Panel(
        f"[bold]Research Question:[/bold]\n{question}",
        title="Starting Research Crew",
        border_style="blue"
    ))
    console.print("\n")

    # Run the crew
    try:
        console.print("[bold]Running research crew...[/bold]\n")
        console.print("[dim]This may take a few minutes. Watch the agent reasoning below.[/dim]\n")
        console.print("=" * 60 + "\n")

        report = run_research(question)

        console.print("\n" + "=" * 60)
        console.print("\n[bold green]Research Complete![/bold green]\n")

        # Display the report
        console.print(Panel(
            Markdown(report),
            title="Literature Review",
            border_style="green"
        ))

        # Save the report
        if not args.no_save:
            filepath = save_report(report, question)
            console.print(f"\n[dim]Report saved to: {filepath}[/dim]")

    except NotImplementedError as e:
        console.print(f"\n[red]Not Implemented: {e}[/red]")
        console.print("\n[yellow]You need to implement the agents, tasks, and crew first![/yellow]")
        console.print("See the TODO comments in:")
        console.print("  - agents/*.py")
        console.print("  - tasks/task_definitions.py")
        console.print("  - crew.py")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error running research crew: {e}[/red]")
        console.print("[dim]Check that your environment is set up correctly.[/dim]")
        console.print("[dim]Run 'python scripts/verify_setup.py' to diagnose issues.[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
