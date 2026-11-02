"""
Research Crew Configuration

Wires the four agents and their tasks into a sequential crew and exposes
run_research() to execute it.
"""

# Load environment variables BEFORE importing crewai
from dotenv import load_dotenv
load_dotenv()

import time

from crewai import Crew, Process
from agents import query_expander, source_hunter, synthesizer, report_writer
from tasks.task_definitions import create_research_tasks
from citations import finalize_report

# Hosted LLMs return temporary overload/rate-limit errors (e.g. Gemini 503/429).
# Retry the whole run a few times with a growing delay instead of failing.
TRANSIENT_ERRORS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 45


def create_research_crew(research_question: str) -> Crew:
    """
    Create a Research Crew configured for the given question.

    Args:
        research_question: The research question to investigate

    Returns:
        Configured Crew ready to execute
    """
    tasks = create_research_tasks(research_question)

    return Crew(
        agents=[query_expander, source_hunter, synthesizer, report_writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        # memory is off: CrewAI memory defaults to OpenAI embeddings, which
        # this setup (Gemini) does not configure; context passing between
        # tasks already carries what each agent needs.
        memory=False,
    )


def run_research(research_question: str) -> str:
    """
    Execute the research crew and return the final report.

    Args:
        research_question: The research question to investigate

    Returns:
        The final literature review as a string
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Rebuild the crew each attempt so no half-finished state carries over.
            crew = create_research_crew(research_question)
            # Authors, years, and titles come from the paper index, not the LLM.
            report, warnings = finalize_report(str(crew.kickoff()))
            for w in warnings:
                print(f"[citations] WARNING: {w}")
            return report
        except Exception as e:
            transient = any(s in str(e) for s in TRANSIENT_ERRORS)
            if not transient or attempt == MAX_ATTEMPTS:
                raise
            wait = BACKOFF_SECONDS * attempt
            print(f"\n[crew] Transient LLM error ({str(e)[:80]}...). "
                  f"Retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})\n")
            time.sleep(wait)


# Allow running crew.py directly for testing
if __name__ == "__main__":
    test_question = "What are the main approaches to building AI agents that can reason and act?"
    print(f"Testing crew with question: {test_question}\n")
    result = run_research(test_question)
    print("\n" + "=" * 50)
    print("FINAL REPORT:")
    print("=" * 50)
    print(result)
