"""
Embedding Agent - Handles document embedding and vector storage
This module manages embedding generation and ChromaDB storage with persistence.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
CHROMA_DB_PATH = Path("data/chroma_db")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "dept_rag"

class EmbeddingManager:
    """Manages embeddings and vector storage"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        persist_directory: Optional[Path] = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME
    ):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Name of the sentence transformer model
            persist_directory: Directory for persistent ChromaDB storage
            collection_name: Name of the ChromaDB collection
        """
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        
        # Setup ChromaDB with persistence
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            self.client = chromadb.Client()
        
        self.collection = None
    
    def create_or_get_collection(self, reset: bool = False) -> chromadb.Collection:
        """
        Create or get the ChromaDB collection
        
        Args:
            reset: Whether to reset the collection if it exists
            
        Returns:
            ChromaDB collection
        """
        if reset and self.collection_name in [col.name for col in self.client.list_collections()]:
            logger.info(f"Resetting collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
        return self.collection
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed a batch of texts with progress tracking
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding documents")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def add_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> chromadb.Collection:
        """
        Add documents to the collection with embeddings
        
        Args:
            documents: List of document texts
            metadata_list: Optional list of metadata dicts
            ids: Optional list of document IDs
            batch_size: Batch size for adding to collection
            
        Returns:
            Updated collection
        """
        if not documents:
            logger.warning("No documents to add")
            return self.collection
        
        # Ensure collection exists
        if self.collection is None:
            self.create_or_get_collection()
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Create default metadata if not provided
        if metadata_list is None:
            metadata_list = [{"index": i} for i in range(len(documents))]
        
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = self.embed_batch(documents, show_progress=True)
        
        # Add to collection in batches
        logger.info(f"Adding {len(documents)} documents to collection...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Storing in ChromaDB"):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadata_list[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        logger.info(f"Successfully added {len(documents)} documents. Total: {self.collection.count()}")
        return self.collection
    
    def update_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadata_list: Optional[List[Dict]] = None
    ):
        """
        Update existing documents in the collection
        
        Args:
            documents: List of document texts
            ids: List of document IDs to update
            metadata_list: Optional list of metadata dicts
        """
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        embeddings = self.embed_batch(documents, show_progress=False)
        
        self.collection.update(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata_list,
            ids=ids
        )
        
        logger.info(f"Updated {len(documents)} documents")
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents from the collection
        
        Args:
            ids: List of document IDs to delete
        """
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def get_collection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        
        return {
            "name": self.collection_name,
            "total_documents": count,
            "model": self.model.get_sentence_embedding_dimension(),
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }
    
    def peek_collection(self, limit: int = 10) -> Dict:
        """
        Peek at some documents in the collection
        
        Args:
            limit: Number of documents to peek at
            
        Returns:
            Sample documents from the collection
        """
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        return self.collection.peek(limit=limit)


# Global embedding manager
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def embed_and_store(chunks: List[str], metadata_list: Optional[List[Dict]] = None) -> chromadb.Collection:
    """
    Embed and store chunks in ChromaDB (backward compatible)
    
    Args:
        chunks: List of text chunks
        metadata_list: Optional list of metadata
        
    Returns:
        ChromaDB collection
    """
    manager = get_embedding_manager()
    manager.create_or_get_collection(reset=False)
    return manager.add_documents(chunks, metadata_list=metadata_list)


def embed_and_store_with_reset(
    chunks: List[str],
    metadata_list: Optional[List[Dict]] = None
) -> chromadb.Collection:
    """
    Embed and store chunks with collection reset
    
    Args:
        chunks: List of text chunks
        metadata_list: Optional list of metadata
        
    Returns:
        ChromaDB collection
    """
    manager = get_embedding_manager()
    manager.create_or_get_collection(reset=True)
    return manager.add_documents(chunks, metadata_list=metadata_list)
