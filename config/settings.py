"""
Central configuration management for Research Crew.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Embedding provider: "openai" or "gemini"
    embedding_provider: Literal["openai", "gemini"]

    # API Keys
    openai_api_key: str | None
    google_api_key: str | None

    # Chunking configuration
    chunk_size: int
    chunk_overlap: int

    # Paths
    vectorstore_path: Path
    papers_path: Path
    processed_path: Path
    evals_path: Path
    outputs_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            vectorstore_path=PROJECT_ROOT / os.getenv("VECTORSTORE_PATH", "data/vectorstore"),
            papers_path=PROJECT_ROOT / os.getenv("PAPERS_PATH", "data/papers"),
            processed_path=PROJECT_ROOT / os.getenv("PROCESSED_PATH", "data/processed"),
            evals_path=PROJECT_ROOT / os.getenv("EVALS_PATH", "data/evals"),
            outputs_path=PROJECT_ROOT / os.getenv("OUTPUTS_PATH", "outputs"),
        )

    def validate(self) -> list[str]:
        """Validate settings and return list of errors."""
        errors = []

        if self.embedding_provider not in ("openai", "gemini"):
            errors.append(f"Invalid EMBEDDING_PROVIDER: {self.embedding_provider}. Must be 'openai' or 'gemini'.")

        if self.embedding_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when EMBEDDING_PROVIDER is 'openai'.")

        if self.embedding_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER is 'gemini'.")

        if self.chunk_size < 100:
            errors.append(f"CHUNK_SIZE ({self.chunk_size}) is too small. Minimum is 100.")

        if self.chunk_overlap >= self.chunk_size:
            errors.append(f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than CHUNK_SIZE ({self.chunk_size}).")

        return errors

    def get_api_key(self) -> str:
        """Get the appropriate API key based on the embedding provider."""
        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            return self.openai_api_key
        else:
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            return self.google_api_key

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.papers_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.evals_path.mkdir(parents=True, exist_ok=True)
        self.outputs_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings.from_env()
