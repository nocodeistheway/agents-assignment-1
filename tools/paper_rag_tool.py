"""
Paper RAG Tool for CrewAI

Provides a search interface over the curated paper corpus
using ChromaDB vector store and embeddings.
"""

import logging
from pathlib import Path

from crewai.tools import tool
import chromadb

from config.settings import settings
from tools.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

# Global vector store client (lazy loaded)
_chroma_client = None
_collection = None
_embedding_provider = None

COLLECTION_NAME = "agentic_ai_papers"


def _get_collection():
    """Get or create the ChromaDB collection."""
    global _chroma_client, _collection, _embedding_provider

    if _collection is not None:
        return _collection

    # Initialize ChromaDB
    vectorstore_path = settings.vectorstore_path

    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Run 'python scripts/setup_vectorstore.py' first."
        )

    _chroma_client = chromadb.PersistentClient(path=str(vectorstore_path))

    # Get the collection
    try:
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run 'python scripts/setup_vectorstore.py' to create it."
        ) from e

    # Initialize embedding provider
    _embedding_provider = get_embedding_provider()

    logger.info(f"Loaded collection with {_collection.count()} chunks")

    return _collection


def _format_results(results: dict, k: int) -> str:
    """Format ChromaDB results into readable output."""
    if not results or not results.get("documents"):
        return "No relevant passages found."

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output_lines = [f"Found {min(len(documents), k)} relevant passages:\n"]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        # Convert distance to similarity score (ChromaDB uses L2 distance)
        # Lower distance = higher similarity
        similarity = max(0, 1 - dist / 2)  # Rough conversion

        paper_id = meta.get("paper_id", "unknown")
        section = meta.get("section", "unknown")
        paper_title = meta.get("paper_title", "unknown")

        output_lines.append(
            f"[{i}] Source: {paper_id} | Section: {section} | Relevance: {similarity:.2f}"
        )
        output_lines.append(f"    Title: {paper_title}")

        # Truncate long documents for display
        doc_display = doc[:500] + "..." if len(doc) > 500 else doc
        output_lines.append(f'    "{doc_display}"')
        output_lines.append("")

    return "\n".join(output_lines)


@tool("Search Research Papers")
def search_papers(query: str, k: int = 5) -> str:
    """
    Search the curated collection of 15 research papers on AI agents.

    Use this to find relevant passages about:
    - Agent architectures and definitions (Wooldridge, surveys)
    - Reasoning patterns (ReAct, Chain-of-Thought, Tree of Thoughts, Reflexion)
    - Multi-agent systems (CAMEL, Generative Agents, AutoGen)
    - Tool use and RAG (Toolformer, RAG papers)
    - Planning and safety (planning abilities, Constitutional AI)

    Args:
        query: Natural language search query describing what you're looking for
        k: Number of results to return (default 5, max 10)

    Returns:
        Relevant passages with paper citations and relevance scores
    """
    global _embedding_provider

    # Validate k
    k = min(max(1, k), 10)

    try:
        collection = _get_collection()
    except Exception as e:
        return f"Error loading vector store: {e}"

    try:
        # Generate query embedding
        query_embedding = _embedding_provider.embed_query(query)

        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        return _format_results(results, k)

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error searching papers: {e}"


def search_papers_raw(query: str, k: int = 5) -> dict:
    """
    Search papers and return raw results (for programmatic use).

    Args:
        query: Search query
        k: Number of results

    Returns:
        Raw ChromaDB results dictionary
    """
    global _embedding_provider

    k = min(max(1, k), 10)
    collection = _get_collection()
    query_embedding = _embedding_provider.embed_query(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )


def get_collection_stats() -> dict:
    """Get statistics about the vector store collection."""
    try:
        collection = _get_collection()
        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": collection.count(),
            "embedding_provider": settings.embedding_provider,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Quick test
    print("Testing paper RAG tool...")

    try:
        stats = get_collection_stats()
        print(f"Collection stats: {stats}")

        query = "What is an intelligent agent?"
        print(f"\nQuery: {query}\n")

        result = search_papers(query, k=3)
        print(result)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set your API key in .env")
        print("2. Run 'python scripts/setup_vectorstore.py'")
