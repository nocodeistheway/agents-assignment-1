"""
Chunking configuration for document processing.

Defines parameters for splitting documents into retrieval-ready chunks.
"""

# Main chunking configuration
CHUNKING_CONFIG = {
    "chunk_size": 800,           # Target tokens per chunk
    "chunk_overlap": 200,        # Overlap between consecutive chunks
    "min_chunk_size": 100,       # Minimum chunk size (avoid tiny chunks)
    "respect_sections": True,    # Try to avoid splitting across sections
    "separators": ["\n\n", "\n", ". ", " "],  # Split hierarchy
}

# Section headers commonly found in academic papers
# Used for section detection and respecting boundaries
SECTION_HEADERS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "methodology",
    "method",
    "methods",
    "approach",
    "proposed method",
    "framework",
    "model",
    "architecture",
    "experiments",
    "experimental setup",
    "experimental results",
    "results",
    "evaluation",
    "analysis",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
    "appendices",
    "supplementary material",
]

# Patterns that indicate we should NOT split (keep together)
KEEP_TOGETHER_PATTERNS = [
    r"Figure \d+",      # Figure references
    r"Table \d+",       # Table references
    r"Equation \d+",    # Equation references
    r"Algorithm \d+",   # Algorithm references
    r"\[\d+\]",         # Citation references
]

# Patterns to clean from extracted text
CLEANUP_PATTERNS = [
    (r"\s+", " "),                    # Multiple spaces -> single space
    (r"\n\s*\n\s*\n+", "\n\n"),      # Multiple blank lines -> double newline
    (r"-\n", ""),                     # Hyphenation at line breaks
    (r"(\w)-\s+(\w)", r"\1\2"),      # Broken words
]
