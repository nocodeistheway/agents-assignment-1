"""
Embedding Provider Abstraction

Supports OpenAI and Gemini embedding models with a unified interface.
Includes batching, rate limiting, and error handling.
"""

import time
import logging
from abc import ABC, abstractmethod

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI text-embedding-3-small implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key (defaults to env var)
            model: Model name
            batch_size: Number of texts per API call
            max_retries: Maximum retry attempts
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self._model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Dimension depends on model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        return self._model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self._model
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"OpenAI API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents with batching."""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

            # Small delay between batches to avoid rate limits
            if i + self.batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed_batch([text])[0]


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embedding-001 implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "models/gemini-embedding-001",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize Gemini embeddings.

        Args:
            api_key: Google API key (defaults to env var)
            model: Model name
            batch_size: Number of texts per API call
            max_retries: Maximum retry attempts
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)
        self.genai = genai
        self._model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    @property
    def dimension(self) -> int:
        return 3072

    @property
    def model_name(self) -> str:
        return self._model

    def _embed_single(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        """Embed a single text with retries."""
        for attempt in range(self.max_retries):
            try:
                result = self.genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type=task_type
                )
                return result['embedding']
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        if not texts:
            return []

        embeddings = []
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                logger.debug(f"Embedded {i}/{len(texts)} documents")

            embedding = self._embed_single(text, task_type="retrieval_document")
            embeddings.append(embedding)

            # Small delay to avoid rate limits
            if i < len(texts) - 1:
                time.sleep(0.05)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed_single(text, task_type="retrieval_query")


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider.

    Args:
        provider: Provider name ("openai" or "gemini"), defaults to settings

    Returns:
        Configured EmbeddingProvider instance
    """
    if provider is None:
        provider = settings.embedding_provider

    provider = provider.lower()

    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "gemini":
        return GeminiEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'gemini'.")


# For convenience, expose a default provider
def get_default_embeddings() -> EmbeddingProvider:
    """Get the default embedding provider based on settings."""
    return get_embedding_provider()


if __name__ == "__main__":
    # Quick test
    print(f"Embedding provider: {settings.embedding_provider}")

    try:
        provider = get_embedding_provider()
        print(f"Model: {provider.model_name}")
        print(f"Dimension: {provider.dimension}")

        # Test embedding
        test_text = "What is an intelligent agent?"
        embedding = provider.embed_query(test_text)
        print(f"Test embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your API key is set in .env file")
