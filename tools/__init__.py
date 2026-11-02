"""
Research Crew Tools

This module contains tools used by the research agents:
- PDF processing for text extraction
- Intelligent document chunking
- Embedding generation (OpenAI/Gemini)
- RAG tool for paper search
"""

from tools.paper_rag_tool import search_papers

__all__ = ["search_papers"]
