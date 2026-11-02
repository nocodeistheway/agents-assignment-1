"""
Research Crew Agents

This module contains the four agents for the Research & Report Crew:
1. QueryExpander - Transforms research questions into search strategies
2. SourceHunter - Searches the paper corpus using RAG
3. Synthesizer - Analyzes and synthesizes findings
4. ReportWriter - Generates the final literature review
"""

# Load environment variables BEFORE importing crewai
from dotenv import load_dotenv
load_dotenv()

from agents.query_expander import query_expander
from agents.source_hunter import source_hunter
from agents.synthesizer import synthesizer
from agents.report_writer import report_writer

__all__ = [
    "query_expander",
    "source_hunter",
    "synthesizer",
    "report_writer"
]
