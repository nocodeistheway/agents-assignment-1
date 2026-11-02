"""
Task Definitions for Research Crew

Four sequential tasks:
1. Query Expansion - Break down the research question
2. Source Hunting - Search the paper corpus
3. Synthesis - Analyze and synthesize findings
4. Report Writing - Generate the literature review

Each task passes its output forward through the context parameter.
"""

from typing import Any, Tuple

from crewai import Task
from agents import query_expander, source_hunter, synthesizer, report_writer
from citations import build_citation_key, load_papers

MIN_DISTINCT_PAPERS = 6
MIN_EXCERPTS = 8


def evidence_coverage_guardrail(output) -> Tuple[bool, Any]:
    """
    Reject a thin evidence log so the Source Hunter searches again.

    Checks the log names enough distinct corpus papers and contains enough
    excerpts. On failure the feedback tells the agent what is missing and
    which papers it has not used yet.
    """
    text = output.raw if hasattr(output, "raw") else str(output)
    papers = load_papers()
    used = [p for p in papers if p["id"] in text]
    excerpts = text.count("Excerpt")
    if len(used) >= MIN_DISTINCT_PAPERS and excerpts >= MIN_EXCERPTS:
        return True, output
    unused = [f"{p['id']} ({p['title']})" for p in papers if p not in used]
    return False, (
        f"Evidence log is too thin: {len(used)} distinct papers and "
        f"{excerpts} excerpts; need at least {MIN_DISTINCT_PAPERS} papers and "
        f"{MIN_EXCERPTS} excerpts. Run more searches, using different wording "
        "and targeting papers not yet used, then return the full evidence "
        "log. Papers not yet used: " + "; ".join(unused)
    )


def create_research_tasks(research_question: str) -> list[Task]:
    """
    Create the task pipeline for a research question.

    Args:
        research_question: The user's research question

    Returns:
        List of 4 tasks in execution order
    """
    citation_key = build_citation_key()

    # =========================================
    # Task 1: Query Expansion
    # =========================================
    expand_task = Task(
        description=(
            f"Research question: \"{research_question}\"\n\n"
            "Plan the search for a literature review on this question. The "
            "searchable corpus is 15 academic papers on AI agents (agent theory, "
            "LLM reasoning, multi-agent systems, tool use and RAG, planning and "
            "safety).\n\n"
            "Produce:\n"
            "1. 4-6 focused sub-questions that together cover the whole "
            "question. Include at least one definitional, one mechanism/how-it-"
            "works, one comparative, and one limitations/challenges sub-question.\n"
            "2. For each sub-question: 4-6 search keywords or phrases written in "
            "the vocabulary academic papers would use, plus 1-2 synonyms or "
            "alternative phrasings.\n"
            "3. For each sub-question: the search angle, i.e. which kind of "
            "passage would answer it (definition, method description, "
            "experimental result, limitation, comparison).\n"
            "4. A priority order: which sub-questions are most central.\n\n"
            "Do not answer the question. Output the plan only."
        ),
        agent=query_expander,
        expected_output=(
            "A numbered search plan in markdown. Each entry has: "
            "'Sub-question', 'Keywords', 'Synonyms', 'Search angle'. Ends with "
            "a one-line priority ordering."
        ),
    )

    # =========================================
    # Task 2: Source Hunting
    # =========================================
    search_task = Task(
        description=(
            f"Research question: \"{research_question}\"\n\n"
            "Using the search plan from the previous task, search the paper "
            "corpus with the 'Search Research Papers' tool.\n\n"
            "Rules:\n"
            "- Run at least one search per sub-question, and rephrase and re-"
            "search when the first results are weak or off-topic. Before each "
            "search state in a sentence what you still need and why; after each "
            "result state what you learned and what to search next.\n"
            "- Use k=5 to k=8 per search.\n"
            "- Read the list of corpus papers below. For each sub-question, "
            "decide which papers by title are most likely to address it, and "
            "run at least one search worded around that paper's topic (for "
            "example its method name or key term) so it is not missed. Do not "
            "rely only on generic queries.\n"
            "- Aim to use evidence from at least 8 different papers overall "
            "when relevant, and note which papers you searched for but found "
            "nothing useful in.\n"
            "- Collect 8-12 distinct, relevant passages in total, spread across "
            "at least 5 different papers where the corpus allows.\n"
            "- Record each passage exactly as the tool returned it: the paper ID "
            "printed after 'Source:', the section, and a short verbatim excerpt "
            "(1-3 sentences). Never invent, paraphrase into a quote, or add a "
            "source the tool did not return.\n"
            "- If the corpus has no good evidence for a sub-question, say so "
            "explicitly instead of filling the gap.\n\n"
            "Valid paper IDs (for checking only; cite only what the tool "
            "returned):\n"
            f"{citation_key}"
        ),
        agent=source_hunter,
        context=[expand_task],
        guardrail=evidence_coverage_guardrail,
        guardrail_max_retries=3,
        expected_output=(
            "An evidence log in markdown, grouped by sub-question. Each entry: "
            "'Paper ID', 'Section', 'Excerpt' (verbatim), and 'Why relevant' "
            "(one sentence). Ends with a 'Coverage notes' list naming any "
            "sub-question with thin or no evidence and the queries tried."
        ),
    )

    # =========================================
    # Task 3: Synthesis
    # =========================================
    synthesis_task = Task(
        description=(
            f"Research question: \"{research_question}\"\n\n"
            "Analyze the evidence log and identify what the literature actually "
            "says. Work only from the passages you were given.\n\n"
            "Produce:\n"
            "1. 3-5 themes. For each theme: a one-sentence claim, the evidence "
            "supporting it (paper IDs plus the key point from each), and a "
            "judgement of how strong the support is (single paper vs several).\n"
            "2. Agreements: where multiple papers converge.\n"
            "3. Tensions and differences: where papers disagree, make different "
            "assumptions, or solve the same problem in different ways, and what "
            "explains the difference.\n"
            "4. Gaps and limitations: what the evidence does not cover, and "
            "limitations the papers themselves acknowledge.\n"
            "5. 2-3 critical observations that go beyond restating the sources "
            "(for example a trade-off, a dependency between approaches, or an "
            "unstated assumption).\n\n"
            "Every claim must carry the paper ID(s) that support it. Do not "
            "introduce papers or facts that are not in the evidence log.\n\n"
            "Citation key (valid papers):\n"
            f"{citation_key}"
        ),
        agent=synthesizer,
        context=[expand_task, search_task],
        expected_output=(
            "A structured analysis in markdown with the headings 'Themes', "
            "'Agreements', 'Tensions and Differences', 'Gaps and Limitations', "
            "and 'Critical Observations'. Each claim is tagged with paper IDs."
        ),
    )

    # =========================================
    # Task 4: Report Writing
    # =========================================
    report_task = Task(
        description=(
            f"Research question: \"{research_question}\"\n\n"
            "Write the final literature review in markdown, using the synthesis "
            "and the evidence log. Required structure, in this order, using "
            "these exact level-2 headings:\n"
            "## Executive Summary  (150-200 words: question, main findings, "
            "key caveat)\n"
            "## Introduction  (why the question matters, scope, the question "
            "restated)\n"
            "## Methodology  (the corpus of 15 papers, retrieval by semantic "
            "search over passages, how themes were derived, and the limits of "
            "a fixed corpus). Describe the corpus in general terms only: do "
            "NOT cite or name individual papers in the Methodology.\n"
            "## Findings  (3-5 level-3 subsections, one per theme; each opens "
            "with the theme's claim, then supports, compares, and qualifies it "
            "with evidence from multiple papers)\n"
            "## Discussion  (agreements, tensions, trade-offs, gaps, and what "
            "this means for building agents)\n"
            "## Conclusion  (direct answer to the question and open problems)\n"
            "## References\n\n"
            "Citation rules:\n"
            "- NEVER cite by paper ID or in backticks (not `autogen_2023`). The "
            "evidence log uses IDs, but the report must use author-year labels.\n"
            "- Cite inline using the exact label from the citation key below, "
            "in parentheses, e.g. (Yao et al., 2023a). Two papers can share "
            "the same authors and year, so the a/b suffix is part of the "
            "label: never drop it.\n"
            "- Every factual claim needs a citation. Do not cite a paper that "
            "does not appear in the evidence or synthesis.\n"
            "- Under '## References', write one line per cited paper in the "
            "form: '- (label) Title [ID: paper_id]', using the exact label, "
            "title, and ID from the key. List exactly the papers cited in the "
            "text. Do NOT write author names there; they are filled in from "
            "the paper index afterwards.\n"
            "- Do not quote at length; paraphrase and synthesize.\n"
            "- Be critical: compare papers, name trade-offs, and state "
            "limitations. Do not just summarize paper by paper.\n"
            "- Only state a trade-off, limitation, or failure mode if the "
            "evidence or synthesis supports it, and cite it. If you are "
            "inferring something the papers do not say, label it explicitly "
            "as 'a plausible implication' and do not attach a citation to it.\n"
            "- Use the citations in the findings; do not cite a paper only in "
            "the Methodology.\n"
            "- Length: roughly 1200-1800 words.\n\n"
            "Citation key (the only citable papers):\n"
            f"{citation_key}"
        ),
        agent=report_writer,
        context=[expand_task, search_task, synthesis_task],
        expected_output=(
            "A complete markdown literature review with the seven required "
            "sections, inline (Author, Year) citations, and a References list "
            "that matches the in-text citations and the citation key."
        ),
    )

    return [expand_task, search_task, synthesis_task, report_task]
