"""
PDF Processor for Research Papers

Extracts text from academic PDFs with section detection.
Handles two-column layouts and common PDF artifacts.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from config.chunking import SECTION_HEADERS, CLEANUP_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """Represents an extracted section from a PDF."""
    paper_id: str
    section_name: str
    text: str
    page_numbers: list[int] = field(default_factory=list)


def clean_extracted_text(text: str) -> str:
    """
    Clean up common PDF extraction artifacts.

    Args:
        text: Raw extracted text from PDF

    Returns:
        Cleaned text
    """
    # Apply cleanup patterns
    for pattern, replacement in CLEANUP_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # Remove excessive whitespace while preserving paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')

    return '\n'.join(cleaned_lines)


def detect_sections(text: str) -> list[tuple[str, int, int]]:
    """
    Detect section boundaries in text.

    Args:
        text: Document text

    Returns:
        List of (section_name, start_idx, end_idx) tuples
    """
    sections = []
    text_lower = text.lower()

    # Find all section headers and their positions
    header_positions = []
    for header in SECTION_HEADERS:
        # Look for headers at start of line or after numbers (e.g., "1. Introduction")
        patterns = [
            rf'^{re.escape(header)}\s*$',  # Header on its own line
            rf'^\d+\.?\s*{re.escape(header)}\s*$',  # Numbered header
            rf'^[ivxIVX]+\.?\s*{re.escape(header)}\s*$',  # Roman numeral header
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                header_positions.append((match.start(), header.title(), match.end()))

    # Sort by position
    header_positions.sort(key=lambda x: x[0])

    # Create sections
    if not header_positions:
        # No sections detected, treat entire text as one section
        return [("Full Text", 0, len(text))]

    for i, (start, name, _) in enumerate(header_positions):
        if i + 1 < len(header_positions):
            end = header_positions[i + 1][0]
        else:
            end = len(text)
        sections.append((name, start, end))

    # Add any text before first section as "Preamble"
    if header_positions and header_positions[0][0] > 0:
        sections.insert(0, ("Preamble", 0, header_positions[0][0]))

    return sections


def extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page.

    Handles common academic paper layouts including two-column formats.

    Args:
        page: PyMuPDF page object

    Returns:
        Extracted text from the page
    """
    # Try to extract with better handling of columns
    # Use "dict" mode for more control
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    # Sort blocks by position (top to bottom, left to right)
    text_blocks = []
    for block in blocks:
        if block["type"] == 0:  # Text block
            bbox = block["bbox"]
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
                text += "\n"
            if text.strip():
                text_blocks.append({
                    "text": text,
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2] - bbox[0]
                })

    # Detect if this is a two-column layout
    page_width = page.rect.width
    mid_point = page_width / 2

    left_blocks = [b for b in text_blocks if b["x"] + b["width"] / 2 < mid_point]
    right_blocks = [b for b in text_blocks if b["x"] + b["width"] / 2 >= mid_point]

    # If roughly equal content on both sides, it's likely two-column
    if left_blocks and right_blocks:
        left_text_len = sum(len(b["text"]) for b in left_blocks)
        right_text_len = sum(len(b["text"]) for b in right_blocks)

        # If both columns have substantial text, process as two-column
        if left_text_len > 100 and right_text_len > 100:
            # Sort each column by y position
            left_blocks.sort(key=lambda b: b["y"])
            right_blocks.sort(key=lambda b: b["y"])

            # Combine: left column first, then right column
            all_text = ""
            for block in left_blocks:
                all_text += block["text"]
            for block in right_blocks:
                all_text += block["text"]
            return all_text

    # Single column: sort by y position
    text_blocks.sort(key=lambda b: (b["y"], b["x"]))
    return "".join(b["text"] for b in text_blocks)


def extract_text_from_pdf(pdf_path: str | Path, paper_id: str) -> list[ExtractedSection]:
    """
    Extract text from a PDF with section detection.

    Args:
        pdf_path: Path to the PDF file
        paper_id: Unique identifier for the paper

    Returns:
        List of ExtractedSection objects
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Extracting text from: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF {pdf_path}: {e}")

    # Extract text from all pages
    full_text = ""
    page_boundaries = [0]  # Track where each page starts in the text

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = extract_text_from_page(page)
        full_text += page_text
        page_boundaries.append(len(full_text))

    doc.close()

    # Clean the extracted text
    full_text = clean_extracted_text(full_text)

    # Detect sections
    sections_info = detect_sections(full_text)

    # Create ExtractedSection objects with page numbers
    sections = []
    for section_name, start_idx, end_idx in sections_info:
        section_text = full_text[start_idx:end_idx].strip()

        if not section_text:
            continue

        # Determine which pages this section spans
        section_pages = []
        for page_num, (page_start, page_end) in enumerate(zip(page_boundaries[:-1], page_boundaries[1:])):
            if page_start < end_idx and page_end > start_idx:
                section_pages.append(page_num + 1)  # 1-indexed

        sections.append(ExtractedSection(
            paper_id=paper_id,
            section_name=section_name,
            text=section_text,
            page_numbers=section_pages
        ))

    logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")

    return sections


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Get basic information about a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with page count, metadata, etc.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    info = {
        "filename": pdf_path.name,
        "page_count": len(doc),
        "metadata": doc.metadata,
        "file_size_kb": pdf_path.stat().st_size / 1024
    }

    doc.close()

    return info


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        sections = extract_text_from_pdf(pdf_path, "test_paper")
        for section in sections:
            print(f"\n{'='*50}")
            print(f"Section: {section.section_name}")
            print(f"Pages: {section.page_numbers}")
            print(f"Text length: {len(section.text)} characters")
            print(section.text[:500] + "..." if len(section.text) > 500 else section.text)
