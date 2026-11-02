"""
Task Definitions for Research Crew

TODO: Define the four sequential tasks:
1. Query Expansion - Break down the research question
2. Source Hunting - Search the paper corpus
3. Synthesis - Analyze and synthesize findings
4. Report Writing - Generate the literature review

Each task should:
- Have a clear description telling the agent what to do
- Specify the agent responsible
- Define expected_output format
- Use context parameter to pass information between tasks
"""

from crewai import Task
from agents import query_expander, source_hunter, synthesizer, report_writer


def create_research_tasks(research_question: str) -> list[Task]:
    """
    Create the task pipeline for a research question.

    Args:
        research_question: The user's research question

    Returns:
        List of 4 tasks in execution order

    TODO: Implement the four tasks below
    """

    # =========================================
    # Task 1: Query Expansion
    # =========================================
    # TODO: Create a task that breaks down the research question
    # into sub-questions, keywords, and search angles
    #
    # expand_task = Task(
    #     description=f"... {research_question} ...",
    #     agent=query_expander,
    #     expected_output="..."
    # )

    # =========================================
    # Task 2: Source Hunting
    # =========================================
    # TODO: Create a task that searches the paper corpus
    # Hint: Use context=[expand_task] to pass the query strategy
    #
    # search_task = Task(
    #     description="...",
    #     agent=source_hunter,
    #     context=[expand_task],
    #     expected_output="..."
    # )

    # =========================================
    # Task 3: Synthesis
    # =========================================
    # TODO: Create a task that synthesizes findings into themes
    # Hint: Use context=[expand_task, search_task] for full context
    #
    # synthesis_task = Task(
    #     description="...",
    #     agent=synthesizer,
    #     context=[expand_task, search_task],
    #     expected_output="..."
    # )

    # =========================================
    # Task 4: Report Writing
    # =========================================
    # TODO: Create a task that writes the final literature review
    # Hint: Use context=[expand_task, search_task, synthesis_task]
    #
    # report_task = Task(
    #     description="...",
    #     agent=report_writer,
    #     context=[expand_task, search_task, synthesis_task],
    #     expected_output="..."
    # )

    # TODO: Return your tasks in order
    # return [expand_task, search_task, synthesis_task, report_task]

    # Placeholder - replace with your implementation
    raise NotImplementedError(
        "TODO: Implement create_research_tasks() in tasks/task_definitions.py"
    )
