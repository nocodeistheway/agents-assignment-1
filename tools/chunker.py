"""
Document Chunker for RAG

Intelligently splits documents into retrieval-ready chunks
with proper overlap and context preservation.
"""

import logging
from dataclasses import dataclass, field

import tiktoken

from config.chunking import CHUNKING_CONFIG

logger = logging.getLogger(__name__)

# Initialize tokenizer for OpenAI models
_tokenizer = None


def get_tokenizer():
    """Get or create the tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


@dataclass
class Chunk:
    """Represents a document chunk ready for embedding."""
    text: str
    paper_id: str
    paper_title: str
    section: str
    page_numbers: list[int] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0

    def to_embedding_text(self) -> str:
        """
        Format chunk for embedding with context.

        Returns:
            Formatted text with metadata context
        """
        pages_str = ", ".join(map(str, self.page_numbers)) if self.page_numbers else "N/A"

        return f"""[Paper: {self.paper_title}]
[Section: {self.section}]
[Pages: {pages_str}]

{self.text}"""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section": self.section,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }


def split_text_by_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:
    """
    Split text using a hierarchy of separators.

    Args:
        text: Text to split
        separators: List of separators in order of preference
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # If text is small enough, return as is
    if count_tokens(text) <= chunk_size:
        return [text.strip()]

    # Try each separator
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            if len(parts) > 1:
                break
    else:
        # No separator found, split by characters
        parts = [text]

    # Combine parts into chunks
    chunks = []
    current_chunk = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Add separator back (except for space)
        if sep != " " and current_chunk:
            test_chunk = current_chunk + sep + part
        else:
            test_chunk = current_chunk + " " + part if current_chunk else part

        if count_tokens(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Current chunk is full
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Check if part itself is too large
            if count_tokens(part) > chunk_size:
                # Recursively split with next separator
                remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else [" "]
                if remaining_seps:
                    sub_chunks = split_text_by_separators(part, remaining_seps, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Last resort: split by character count
                    tokenizer = get_tokenizer()
                    tokens = tokenizer.encode(part)
                    for i in range(0, len(tokens), chunk_size - chunk_overlap):
                        chunk_tokens = tokens[i:i + chunk_size]
                        chunks.append(tokenizer.decode(chunk_tokens))
                    current_chunk = ""
            else:
                current_chunk = part

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Get overlap from previous chunk
                prev_tokens = get_tokenizer().encode(chunks[i - 1])
                overlap_tokens = prev_tokens[-chunk_overlap:] if len(prev_tokens) > chunk_overlap else prev_tokens
                overlap_text = get_tokenizer().decode(overlap_tokens)
                chunk = overlap_text + " " + chunk
            overlapped_chunks.append(chunk.strip())
        chunks = overlapped_chunks

    return chunks


def chunk_document(
    text: str,
    paper_id: str,
    paper_title: str,
    section_name: str,
    page_numbers: list[int],
    config: dict | None = None
) -> list[Chunk]:
    """
    Chunk a document section into retrieval-ready pieces.

    Args:
        text: Section text to chunk
        paper_id: Paper identifier
        paper_title: Paper title
        section_name: Name of the section
        page_numbers: Page numbers the section spans
        config: Optional chunking configuration override

    Returns:
        List of Chunk objects
    """
    if config is None:
        config = CHUNKING_CONFIG

    chunk_size = config.get("chunk_size", 800)
    chunk_overlap = config.get("chunk_overlap", 200)
    min_chunk_size = config.get("min_chunk_size", 100)
    separators = config.get("separators", ["\n\n", "\n", ". ", " "])

    # Split text into chunks
    text_chunks = split_text_by_separators(text, separators, chunk_size, chunk_overlap)

    # Filter out tiny chunks and create Chunk objects
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        token_count = count_tokens(chunk_text)

        # Skip chunks that are too small (unless it's the only chunk)
        if token_count < min_chunk_size and len(text_chunks) > 1:
            # Try to merge with previous chunk if possible
            if chunks and count_tokens(chunks[-1].text + " " + chunk_text) <= chunk_size:
                chunks[-1].text += " " + chunk_text
                chunks[-1].token_count = count_tokens(chunks[-1].text)
                continue

        chunks.append(Chunk(
            text=chunk_text,
            paper_id=paper_id,
            paper_title=paper_title,
            section=section_name,
            page_numbers=page_numbers.copy(),
            chunk_index=i,
            token_count=token_count
        ))

    # Re-number chunks after any merging
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


def chunk_sections(
    sections: list,
    paper_title: str,
    config: dict | None = None
) -> list[Chunk]:
    """
    Chunk multiple sections from a paper.

    Args:
        sections: List of ExtractedSection objects
        paper_title: Title of the paper
        config: Optional chunking configuration

    Returns:
        List of all Chunk objects
    """
    all_chunks = []

    for section in sections:
        section_chunks = chunk_document(
            text=section.text,
            paper_id=section.paper_id,
            paper_title=paper_title,
            section_name=section.section_name,
            page_numbers=section.page_numbers,
            config=config
        )
        all_chunks.extend(section_chunks)

    # Re-number all chunks globally
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    logger.info(f"Created {len(all_chunks)} chunks for paper: {paper_title}")

    return all_chunks


def estimate_embedding_cost(chunks: list[Chunk], model: str = "text-embedding-3-small") -> dict:
    """
    Estimate the cost of embedding chunks.

    Args:
        chunks: List of chunks to embed
        model: Embedding model name

    Returns:
        Dictionary with cost estimates
    """
    total_tokens = sum(count_tokens(chunk.to_embedding_text()) for chunk in chunks)

    # Pricing as of 2024 (per 1M tokens)
    pricing = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-004": 0.00,  # Gemini is free tier
    }

    price_per_million = pricing.get(model, 0.02)
    estimated_cost = (total_tokens / 1_000_000) * price_per_million

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "model": model,
        "estimated_cost_usd": round(estimated_cost, 4)
    }


if __name__ == "__main__":
    # Quick test
    test_text = """
    Abstract

    This is a test abstract with some important content about AI agents.
    It discusses various approaches to building autonomous systems.

    Introduction

    The field of artificial intelligence has seen remarkable progress in recent years.
    Large language models have demonstrated impressive capabilities across a wide range
    of tasks. This paper explores how these models can be used to build autonomous agents.

    We present a novel framework for agent construction that leverages the reasoning
    capabilities of LLMs while maintaining safety and alignment with human values.
    """

    chunks = chunk_document(
        text=test_text,
        paper_id="test_paper",
        paper_title="Test Paper Title",
        section_name="Full Text",
        page_numbers=[1, 2]
    )

    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_index} ({chunk.token_count} tokens) ---")
        print(chunk.to_embedding_text()[:300] + "...")
