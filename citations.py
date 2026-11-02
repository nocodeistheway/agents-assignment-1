"""
Citation helpers built from data/papers/paper_index.json.

The index is the single source of truth for authors, years, titles, and IDs.
LLM agents are given a citation key to cite from, but the final References
section is regenerated here from the index so it can never contain invented
authors or titles.
"""

import json
import re
from collections import Counter
from pathlib import Path

PAPER_INDEX_PATH = Path(__file__).resolve().parent / "data" / "papers" / "paper_index.json"
MAX_LISTED_AUTHORS = 6


def load_papers() -> list[dict]:
    return json.loads(PAPER_INDEX_PATH.read_text())["papers"]


def _short_author(authors: list[str]) -> str:
    """'Surname' / 'Surname & Surname' / 'Surname et al.'"""
    surnames = [a.split()[-1] for a in authors]
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}"
    return f"{surnames[0]} et al."


def citation_labels(papers: list[dict] | None = None) -> dict[str, str]:
    """
    Map paper ID -> in-text label, e.g. 'Yao et al., 2023a'.
    Papers sharing the same author-year get a, b, ... suffixes (index order).
    """
    papers = papers or load_papers()
    base = {p["id"]: f"{_short_author(p['authors'])}, {p['year']}" for p in papers}
    counts = Counter(base.values())
    seen: Counter = Counter()
    labels = {}
    for p in papers:
        label = base[p["id"]]
        if counts[label] > 1:
            suffix = "abcdefghijklmnopqrstuvwxyz"[seen[label]]
            seen[label] += 1
            label = f"{label}{suffix}"
        labels[p["id"]] = label
    return labels


def build_citation_key() -> str:
    """Text block listing every citable paper with its exact label, ID, and title."""
    papers = load_papers()
    labels = citation_labels(papers)
    return "\n".join(
        f"- ({labels[p['id']]}) | ID: {p['id']} | Title: {p['title']}" for p in papers
    )


def _format_author(name: str) -> str:
    parts = name.split()
    initials = " ".join(f"{part[0]}." for part in parts[:-1])
    return f"{parts[-1]}, {initials}".strip().rstrip(",")


def render_reference(paper: dict) -> str:
    authors = [_format_author(a) for a in paper["authors"][:MAX_LISTED_AUTHORS]]
    if len(paper["authors"]) > MAX_LISTED_AUTHORS:
        authors.append("et al.")
    author_str = ", ".join(authors)
    venue = f" {paper['venue']}." if paper.get("venue") else ""
    arxiv = f" arXiv:{paper['arxiv_id']}." if paper.get("arxiv_id") else ""
    return f"{author_str} ({paper['year']}). {paper['title']}.{venue}{arxiv} [ID: {paper['id']}]"


def _citation_pattern(paper: dict, label: str) -> tuple[re.Pattern, re.Pattern | None]:
    """
    Regexes that find a paper's in-text citations, in either style:
    '(Yao et al., 2023a)' or 'Yao et al. (2023a)', 'Wooldridge and Jennings (1995)'.
    The second regex (only for suffixed labels) finds the bare year without the
    a/b suffix, which is ambiguous.
    """
    surnames = [a.split()[-1] for a in paper["authors"]]
    lead = re.escape(surnames[0])
    if len(surnames) == 2:
        lead += r"(?:\s+(?:&|and)\s+" + re.escape(surnames[1]) + r")?"
    elif len(surnames) > 2:
        lead += r"(?:\s+et al\.)?"
    year_label = label.rsplit(",", 1)[1].strip()  # e.g. '2023a'
    tail = r",?\s*\(?\s*"
    exact = re.compile(rf"\b{lead}{tail}{re.escape(year_label)}(?![0-9A-Za-z_])")
    bare = None
    if year_label != str(paper["year"]):
        bare = re.compile(rf"\b{lead}{tail}{paper['year']}(?![0-9A-Za-z_])")
    return exact, bare


def _ids_to_labels(body: str, labels: dict[str, str]) -> str:
    """
    Convert ID-style citations to author-year labels:
    '(`autogen_2023`, `camel_2023`)' -> '(Wu et al., 2023; Li et al., 2023)',
    and any remaining '`autogen_2023`' / 'autogen_2023' -> 'Wu et al., 2023'.
    """
    ids = sorted(labels, key=len, reverse=True)
    id_alt = "|".join(re.escape(i) for i in ids)
    token = rf"`?(?:{id_alt})`?"

    def group(m: re.Match) -> str:
        found = re.findall(rf"(?:{id_alt})", m.group(1))
        return "(" + "; ".join(labels[i] for i in found) + ")"

    body = re.sub(rf"\(\s*({token}(?:\s*[,;]\s*{token})*)\s*\)", group, body)
    return re.sub(rf"`?\b({id_alt})\b`?", lambda m: labels[m.group(1)], body)


def _tidy_parentheses(body: str) -> str:
    """'((A, 2023))' -> '(A, 2023)'; '((A, 2023), (B, 2022))' -> '(A, 2023; B, 2022)'."""
    body = re.sub(
        r"\(((?:\([^()]*\))(?:\s*[,;]\s*\([^()]*\))*)\)",
        lambda m: "(" + "; ".join(re.findall(r"\(([^()]*)\)", m.group(1))) + ")",
        body,
    )
    return body


def finalize_report(report: str) -> tuple[str, list[str]]:
    """
    Make the report's citations consistent with the paper index.

    - Converts ID-style citations (`autogen_2023`) to author-year labels and
      strips leaked ID suffixes from labels ('2023_survey' -> '2023').
    - Replaces the LLM-written '## References' with entries generated from the
      index for exactly the papers cited in the body, in order of first citation.
    - Returns (report, warnings); warnings list ambiguous citations, e.g. a bare
      'Yao et al., 2023' where two papers share author and year.
    """
    warnings: list[str] = []
    match = re.search(r"^## References\s*$", report, flags=re.MULTILINE)
    body = report[: match.start()].rstrip() if match else report.rstrip()
    papers = load_papers()
    labels = citation_labels(papers)
    body = _ids_to_labels(body, labels)
    body = re.sub(r"(\b\d{4}[a-z]?)_[a-z_]+", r"\1", body)
    body = _tidy_parentheses(body)
    found = []  # (first position, paper)
    for paper in papers:
        exact, bare = _citation_pattern(paper, labels[paper["id"]])
        hit = exact.search(body)
        if hit:
            found.append((hit.start(), paper))
        if bare:
            for m in bare.finditer(body):
                warnings.append(
                    f"Ambiguous citation '{m.group(0)}' (needs {labels[paper['id']]})"
                )
    if not found:
        warnings.append("No recognizable citations found; References left as written.")
        return report, warnings

    found.sort(key=lambda x: x[0])
    entries = [
        f"- **({labels[p['id']]})** {render_reference(p)}" for _, p in found
    ]
    return body + "\n\n## References\n\n" + "\n".join(entries) + "\n", warnings
