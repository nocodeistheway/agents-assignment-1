"""
Source Hunter Agent

Searches the curated paper corpus with the search_papers RAG tool and collects
verbatim evidence for every sub-question in the query strategy.
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent
from tools.paper_rag_tool import search_papers

source_hunter = Agent(
    role="Corpus Investigator",
    goal=(
        "For every sub-question in the search strategy, run multiple targeted "
        "searches against the paper corpus and collect 8-12 distinct, relevant "
        "passages, each recorded with its paper ID, section, and a short "
        "verbatim excerpt."
    ),
    backstory=(
        "You are a meticulous investigative researcher who treats the first "
        "search result as a lead, never as an answer. You reason out loud "
        "before each search (what do I still need, which wording is the paper "
        "likely to use), call the search tool, read what came back, and decide "
        "whether to rephrase, broaden, or narrow the next query. You keep "
        "searching until each sub-question has evidence from at least two "
        "different papers, or until you can state clearly that the corpus is "
        "silent on it. You only report what the search tool actually returned: "
        "you never fill gaps from memory, and you always record the exact "
        "paper ID the tool printed next to each passage."
    ),
    tools=[search_papers],
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=25,
)
