"""
Configuration Module - Centralized configuration management
This module provides configuration settings for the entire application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class"""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Database Configuration
    CHROMA_DB_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", str(CHROMA_DB_DIR)))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "dept_rag")
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    
    # Retrieval Configuration
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "10"))
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    
    # LLM Configuration
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # Application Configuration
    APP_TITLE: str = os.getenv("APP_TITLE", "IITB EE RAG Bot")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Paths
    DATA_DIR: Path = DATA_DIR
    RAW_DATA_DIR: Path = RAW_DATA_DIR
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY not set")
        
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP cannot be negative")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if cls.RETRIEVAL_K <= 0:
            errors.append("RETRIEVAL_K must be positive")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Reranker Model: {cls.RERANKER_MODEL}")
        print(f"Use Reranker: {cls.USE_RERANKER}")
        print(f"Collection Name: {cls.COLLECTION_NAME}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Chunking Strategy: {cls.CHUNKING_STRATEGY}")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Rerank Top K: {cls.RERANK_TOP_K}")
        print(f"LLM Temperature: {cls.LLM_TEMPERATURE}")
        print(f"LLM Max Tokens: {cls.LLM_MAX_TOKENS}")
        print(f"ChromaDB Path: {cls.CHROMA_DB_PATH}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 60)


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    LOG_LEVEL: str = "DEBUG"
    USE_RERANKER: bool = True


class ProductionConfig(Config):
    """Production-specific configuration"""
    LOG_LEVEL: str = "WARNING"
    USE_RERANKER: bool = True


class TestConfig(Config):
    """Test-specific configuration"""
    LOG_LEVEL: str = "DEBUG"
    CHUNK_SIZE: int = 200
    RETRIEVAL_K: int = 3
    USE_RERANKER: bool = False


def get_config(env: Optional[str] = None) -> Config:
    """
    Get configuration based on environment
    
    Args:
        env: Environment name (development, production, test)
        
    Returns:
        Configuration object
    """
    env = env or os.getenv("ENV", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig
    }
    
    return configs.get(env, Config)


# Default configuration
config = Config()

# Validate on import
if not config.validate():
    print("\n⚠️  Warning: Configuration validation failed!")
    print("Some features may not work correctly.\n")
