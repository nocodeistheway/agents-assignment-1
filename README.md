# Research Crew Starter Kit

**UCLA Extension -- Agentic AI Course | Week 1 Assignment**

Build a multi-agent Research & Report Crew that searches a curated corpus of 15 foundational AI agent papers and produces structured literature reviews.

---

## What Is This Assignment?

In this assignment you will implement a **4-agent research crew** using the CrewAI framework. The crew takes a research question as input and produces a literature review as output, drawing on a curated corpus of 15 academic papers about AI agents.

The infrastructure is provided for you -- PDF processing, vector store, embeddings, RAG tool, and evaluation suite. **Your job is to design and implement the agents, tasks, and crew configuration** that tie it all together.

### What You Will Build

| Component | File(s) | What You Do |
|-----------|---------|-------------|
| **4 Agents** | `agents/*.py` | Define roles, goals, backstories, and tool assignments |
| **4 Tasks** | `tasks/task_definitions.py` | Write task descriptions, expected outputs, and context dependencies |
| **Crew** | `crew.py` | Wire agents and tasks into a sequential crew pipeline |

### What Is Provided (Do Not Modify)

| Component | Description |
|-----------|-------------|
| `tools/` | PDF processor, chunker, embeddings, RAG search tool |
| `scripts/` | Paper download, vector store setup, verification |
| `evals/` | Retrieval metrics and evaluation runner |
| `config/` | Settings and chunking configuration |
| `main.py` | CLI entry point |
| `data/papers/` | 15 PDF papers + metadata index |

### How the Crew Works

```
Research Question
       |
       v
[Query Expander]  -- Breaks the question into sub-questions and keywords
       |
       v
[Source Hunter]   -- Searches the paper corpus via RAG tool
       |
       v
[Synthesizer]     -- Identifies themes, debates, and gaps across sources
       |
       v
[Report Writer]   -- Produces a structured literature review with citations
       |
       v
Literature Review (saved to outputs/)
```

Each agent receives the output of the previous agent(s) as context. The Source Hunter is the only agent that uses a tool (`search_papers`) to query the vector store.

---

## Paper Corpus

The kit includes 15 foundational papers on agentic AI:

| Category | Papers |
|----------|--------|
| **Agent Theory** | Wooldridge 1995, Wang 2023 Survey, Xi 2023 Survey |
| **Reasoning** | ReAct, Chain-of-Thought, Tree of Thoughts, Reflexion |
| **Multi-Agent** | CAMEL, Generative Agents, AutoGen |
| **Tool Use & RAG** | Toolformer, RAG (2020), RAG Survey (2023) |
| **Planning & Safety** | Planning Abilities, Constitutional AI |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) -- fast Python package manager
- OpenAI API key OR Google API key (for embeddings)
- ~500MB disk space for PDFs and vector store

---

## Getting Started

### 1. Create Your Repository

Go to the GitHub repo and click **"Use this template" > "Create a new repository"**. Name it `agents-assignment-1`, set it to **Public**, and create it. Then clone your new repo:

```bash
git clone https://github.com/<your-username>/agents-assignment-1.git
cd agents-assignment-1
```

> **Why a template?** This creates a clean copy under your account with its own history. You will submit by sharing this repository link.

### 2. Install uv (if you don't have it)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 3. Install Dependencies

```bash
# uv reads pyproject.toml automatically
uv sync
```

> **Alternative (pip):** If you prefer pip, run `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

### 4. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

Choose your embedding provider in `.env`:
```bash
# For OpenAI embeddings
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# OR for Gemini embeddings (free tier available)
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=AIza-your-key-here
```

### 5. Build the Vector Store

```bash
# Build the vector store from the included papers
uv run python scripts/setup_vectorstore.py

# Verify everything works
uv run python scripts/verify_setup.py
```

> **Note:** If you need to re-download papers, run `uv run python scripts/download_papers.py`

### 6. Run the Crew

```bash
# Interactive mode
uv run python main.py -i

# With a specific question
uv run python main.py "What are the main approaches to building AI agents?"

# With an example question
uv run python main.py --example
```

---

## Project Structure

```
research_crew_starter/
├── config/                 # Configuration
│   ├── settings.py        # Environment settings
│   └── chunking.py        # Chunking parameters
├── data/
│   ├── papers/            # PDF files + paper_index.json
│   ├── processed/         # Extracted text cache
│   ├── vectorstore/       # ChromaDB (gitignored)
│   └── evals/             # Test queries
├── tools/                 # Core tools (provided)
│   ├── pdf_processor.py   # PDF text extraction
│   ├── chunker.py         # Document chunking
│   ├── embeddings.py      # OpenAI/Gemini embeddings
│   └── paper_rag_tool.py  # CrewAI search tool
├── scripts/               # Setup scripts (provided)
│   ├── download_papers.py
│   ├── setup_vectorstore.py
│   └── verify_setup.py
├── evals/                 # Evaluation suite (provided)
│   ├── retrieval_metrics.py
│   ├── run_evals.py
│   └── report_generator.py
├── agents/                # <-- YOUR WORK: Agent definitions
│   ├── query_expander.py
│   ├── source_hunter.py
│   ├── synthesizer.py
│   └── report_writer.py
├── tasks/                 # <-- YOUR WORK: Task definitions
│   └── task_definitions.py
├── outputs/               # Generated reports
├── crew.py               # <-- YOUR WORK: Crew configuration
├── main.py               # Entry point (provided)
└── pyproject.toml        # Dependencies
```

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | `"openai"` or `"gemini"` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | -- |
| `GOOGLE_API_KEY` | Google API key | -- |
| `CHUNK_SIZE` | Tokens per chunk | `800` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

---

## What You Are Expected to Deliver

### 1. Working Crew (code)

Your crew must run end-to-end without errors:

```bash
uv run python main.py --example
```

It should produce a literature review that:
- Uses content retrieved from the paper corpus (not fabricated)
- Has clear structure (introduction, findings by theme, conclusion, references)
- Cites source papers properly

### 2. Two Literature Reviews

Run your crew on **two different research questions** and save the outputs to `outputs/`. Example questions:

```bash
uv run python main.py "What are the main approaches to building AI agents that can reason and act?"
uv run python main.py "How do multi-agent systems coordinate and communicate?"
```

### 3. Reflection Document

Write a 1-2 page reflection (PDF) covering:
- How you designed your agents and why
- Examples of ReAct-style reasoning in your agent outputs
- What worked, what didn't, and what you would change
- What would be needed to use this system in production

---

## Submission

Submit your work by pushing to your forked GitHub repository and sharing the link.

### Before You Submit

```bash
# Make sure your crew runs without errors
uv run python main.py --example

# Verify setup is intact
uv run python scripts/verify_setup.py
```

### What Your Repo Should Contain

```
your-forked-repo/
├── agents/                    # Your agent implementations
│   ├── query_expander.py
│   ├── source_hunter.py
│   ├── synthesizer.py
│   └── report_writer.py
├── tasks/
│   └── task_definitions.py    # Your task definitions
├── crew.py                    # Your crew configuration
├── outputs/
│   ├── report_<question1>_<timestamp>.md
│   └── report_<question2>_<timestamp>.md
├── reflection.pdf             # Your 1-2 page reflection
└── ... (all other starter kit files)
```

### How to Submit

1. Commit and push all your changes:
   ```bash
   git add -A
   git commit -m "Complete assignment: Research Crew implementation"
   git push origin main
   ```

2. Verify your repo is accessible -- visit `https://github.com/<your-username>/agents-assignment-1` and confirm your changes are visible.

3. Submit the link to your GitHub repository through the course portal.

> **Important:** Make sure your repo is **public** (or that the instructor has access) and that `outputs/` contains your two generated literature reviews and `reflection.pdf` is included.

---

## Grading Rubric

Your work is evaluated across three categories. Each category is scored on a 4-point scale.

| Score | Label | Meaning |
|-------|-------|---------|
| 4 | Excellent | Exceeds expectations, demonstrates mastery |
| 3 | Proficient | Meets all expectations, solid work |
| 2 | Developing | Partially meets expectations, gaps present |
| 1 | Beginning | Minimal effort or significant issues |

### Agent & Task Design (40%)

| Score | Criteria |
|-------|----------|
| **4** | Agents have distinct, well-crafted roles with specific goals and backstories that meaningfully shape behavior. Tasks have detailed descriptions with clear formatting requirements. Context dependencies are correctly wired. |
| **3** | Agents have clear roles and reasonable goals/backstories. Tasks are well-described. Context passes correctly between tasks. |
| **2** | Agents are defined but roles overlap or goals are vague. Task descriptions are generic. Some context issues. |
| **1** | Agents are minimal stubs. Tasks lack descriptions or expected outputs. Pipeline is broken. |

### Output Quality (40%)

| Score | Criteria |
|-------|----------|
| **4** | Report is well-structured with all sections (summary, introduction, methodology, findings by theme, discussion, conclusion, references). Citations are accurate. Synthesis shows critical analysis, not just summarization. Content is grounded in the paper corpus. |
| **3** | Report has recognizable structure with most sections. Includes citations referencing source papers. Findings are organized by theme. |
| **2** | Report has basic structure but sections are thin or missing. Some RAG content present but citations are sparse or inaccurate. |
| **1** | Output is unstructured text with no citations. Little or no content from the paper corpus. |

### Reflection & Code Quality (20%)

| Score | Criteria |
|-------|----------|
| **4** | Reflection identifies specific ReAct examples in agent output. Discusses design tradeoffs and alternatives considered. Thoughtful production considerations. Code is clean and well-organized. |
| **3** | Reflection covers design decisions and lessons learned. Mentions ReAct patterns. Code runs cleanly. |
| **2** | Reflection is superficial. Limited connection to course concepts. Code has minor issues. |
| **1** | No reflection or purely descriptive. Code has errors or is disorganized. |

### Grade Mapping

| Grade | Typical Profile |
|-------|----------------|
| **A** | Scores 4 in at least two categories, no score below 3 |
| **B** | Scores 3 in all categories, or 4 in one with a 2 elsewhere |
| **C** | Scores 2 in most categories |
| **F** | Scores 1 in most categories or crew does not run |

---

## Running Evaluations

The evaluation suite tests the RAG retrieval quality (not your agent implementation):

```bash
# Run full evaluation suite
uv run python -m evals.run_evals

# With verbose output
uv run python -m evals.run_evals -v
```

Metrics reported: Recall@k, Precision@k, MRR, Concept Coverage.

---

## Example Research Questions

### Beginner
```bash
uv run python main.py "What is an AI agent?"
uv run python main.py "How does chain-of-thought prompting work?"
```

### Intermediate
```bash
uv run python main.py "What are the main approaches to building AI agents that can reason and act?"
uv run python main.py "How do multi-agent systems coordinate and communicate?"
```

### Advanced
```bash
uv run python main.py "Compare and contrast different reasoning frameworks for LLM agents."
uv run python main.py "What are the key challenges in making AI agents safe and aligned?"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Vector store not found" | Run `uv run python scripts/setup_vectorstore.py` |
| "API key not found" | Check your `.env` file has the correct key |
| "PDF download failed" | Some papers may need manual download -- see `data/papers/paper_index.json` for URLs |
| "ChromaDB collection not found" | Delete `data/vectorstore/` and re-run setup |
| Import errors | Make sure you're in the `research_crew_starter/` directory |
| `uv` not found | Install with `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |

---

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [uv Documentation](https://docs.astral.sh/uv/)

---

**UCLA Extension -- Agentic AI Course**
