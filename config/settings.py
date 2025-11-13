"""
Configuration Settings for RAG System
All configurable parameters in one place for easy tuning
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Centralized configuration for RAG system
    All values can be overridden via environment variables
    """

    # ========== API Keys ==========
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")

    # ========== Pinecone Configuration ==========
    PINECONE_INDEX_NAME: str = Field(default="hr-policy-rag", env="PINECONE_INDEX_NAME")
    PINECONE_NAMESPACE: Optional[str] = Field(default=None, env="PINECONE_NAMESPACE")

    # ========== Embedding Model Configuration ==========
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # ========== LLM Configuration ==========
    LLM_MODEL: str = Field(default="gemini-2.5-flash", env="LLM_MODEL")
    LLM_TEMPERATURE: float = Field(default=0.3, env="LLM_TEMPERATURE")
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=1024, env="LLM_MAX_OUTPUT_TOKENS")
    LLM_TOP_P: float = Field(default=0.95, env="LLM_TOP_P")
    LLM_TOP_K: int = Field(default=40, env="LLM_TOP_K")

    # ========== Retrieval Configuration ==========
    RETRIEVAL_TOP_K: int = Field(default=5, env="RETRIEVAL_TOP_K")
    RETRIEVAL_ALPHA: float = Field(default=0.5, env="RETRIEVAL_ALPHA")  # 0.5 = balanced hybrid

    # ========== Memory Configuration ==========
    MEMORY_MAX_PAIRS: int = Field(default=5, env="MEMORY_MAX_PAIRS")  # Last 5 Q&A pairs
    SESSION_TIMEOUT_MINUTES: int = Field(default=30, env="SESSION_TIMEOUT_MINUTES")

    # ========== FastAPI Configuration ==========
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8001, env="API_PORT")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")

    # ========== CORS Configuration ==========
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:8001", "http://127.0.0.1:8001"],
        env="CORS_ORIGINS"
    )

    # ========== BM25 Configuration ==========
    BM25_K1: float = Field(default=1.5, env="BM25_K1")
    BM25_B: float = Field(default=0.75, env="BM25_B")

    # ========== Prompt Templates ==========
    SYSTEM_PROMPT: str = """You are a helpful HR Policy Assistant for a company.
Your role is to answer questions about company policies accurately and professionally.

INSTRUCTIONS:
1. Use ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say so honestly
3. Always cite the relevant section and page numbers from the policy document
4. Be concise but complete in your answers
5. If referring to previous conversation, acknowledge it naturally
6. Maintain a professional and friendly tone

CONTEXT (Retrieved from HR Policy Document):
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

YOUR RESPONSE:"""

    NO_CONTEXT_RESPONSE: str = "I apologize, but I couldn't find relevant information in the HR policy document to answer your question. Could you please rephrase your question or ask about a specific policy topic?"

    ERROR_RESPONSE: str = "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists."

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get settings instance (for dependency injection in FastAPI)
    """
    return settings


# ========== Validation Functions ==========

def validate_settings():
    """
    Validate critical settings on startup
    Raises ValueError if configuration is invalid
    """
    errors = []

    # Check API keys
    if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY == "your_pinecone_api_key_here":
        errors.append("PINECONE_API_KEY is not set or invalid")

    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
        errors.append("GEMINI_API_KEY is not set or invalid")

    # Check ranges
    if not 0.0 <= settings.LLM_TEMPERATURE <= 2.0:
        errors.append("LLM_TEMPERATURE must be between 0.0 and 2.0")

    if not 0.0 <= settings.RETRIEVAL_ALPHA <= 1.0:
        errors.append("RETRIEVAL_ALPHA must be between 0.0 and 1.0")

    if settings.RETRIEVAL_TOP_K < 1:
        errors.append("RETRIEVAL_TOP_K must be at least 1")

    if settings.MEMORY_MAX_PAIRS < 1:
        errors.append("MEMORY_MAX_PAIRS must be at least 1")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


if __name__ == "__main__":
    # Test configuration loading
    print("=" * 80)
    print("  CONFIGURATION VALIDATION")
    print("=" * 80)
    print()

    try:
        validate_settings()
        print("[OK] Configuration validated successfully")
        print()
        print("Current Configuration:")
        print(f"  Pinecone Index: {settings.PINECONE_INDEX_NAME}")
        print(f"  LLM Model: {settings.LLM_MODEL}")
        print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"  Retrieval Top-K: {settings.RETRIEVAL_TOP_K}")
        print(f"  Memory Max Pairs: {settings.MEMORY_MAX_PAIRS}")
        print(f"  API Port: {settings.API_PORT}")
    except ValueError as e:
        print(f"[ERROR] {e}")
