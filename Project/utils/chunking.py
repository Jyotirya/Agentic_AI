"""
Chunking Utilities - Advanced document chunking strategies
This module provides multiple chunking strategies for document preprocessing.
"""

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from typing import List, Dict, Tuple, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def chunk(self, documents: List[str], metadata_list: Optional[List[Dict]] = None) -> Tuple[List[str], List[Dict]]:
        """
        Chunk documents and preserve metadata
        
        Args:
            documents: List of documents to chunk
            metadata_list: Optional list of metadata dicts
            
        Returns:
            Tuple of (chunks, chunk_metadata)
        """
        raise NotImplementedError


class RecursiveChunking(ChunkingStrategy):
    """Recursive character-based chunking"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive chunking
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separator strings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def chunk(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Chunk documents recursively"""
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            chunks = self.splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            # Preserve and extend metadata
            base_metadata = metadata_list[idx] if metadata_list else {}
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source_doc_index": idx,
                    "chunk_length": len(chunk)
                }
                all_metadata.append(chunk_meta)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks, all_metadata


class SemanticChunking(ChunkingStrategy):
    """Semantic-aware chunking based on sentence boundaries"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize semantic chunking
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Chunk documents by semantic units (sentences)"""
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            
            current_chunk = []
            current_length = 0
            chunks = []
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add last chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            all_chunks.extend(chunks)
            
            # Add metadata
            base_metadata = metadata_list[idx] if metadata_list else {}
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source_doc_index": idx,
                    "chunk_length": len(chunk),
                    "chunking_strategy": "semantic"
                }
                all_metadata.append(chunk_meta)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
        return all_chunks, all_metadata


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize fixed-size chunking
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Chunk documents with fixed size"""
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            chunks = []
            start = 0
            
            while start < len(doc):
                end = min(start + self.chunk_size, len(doc))
                chunk = doc[start:end]
                chunks.append(chunk)
                start = end - self.chunk_overlap
                
                if start >= len(doc):
                    break
            
            all_chunks.extend(chunks)
            
            # Add metadata
            base_metadata = metadata_list[idx] if metadata_list else {}
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source_doc_index": idx,
                    "chunk_length": len(chunk),
                    "chunking_strategy": "fixed_size"
                }
                all_metadata.append(chunk_meta)
        
        logger.info(f"Created {len(all_chunks)} fixed-size chunks from {len(documents)} documents")
        return all_chunks, all_metadata


class AdaptiveChunking(ChunkingStrategy):
    """Adaptive chunking that adjusts based on content structure"""
    
    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
        target_chunk_size: int = 500
    ):
        """
        Initialize adaptive chunking
        
        Args:
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            target_chunk_size: Target chunk size
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
    
    def _find_split_point(self, text: str, target: int) -> int:
        """Find optimal split point near target"""
        # Look for natural boundaries near target
        boundaries = ['.', '!', '?', '\n', ',', ';']
        
        # Search within a window around target
        window = 100
        start = max(0, target - window)
        end = min(len(text), target + window)
        
        best_pos = target
        best_priority = -1
        
        for i in range(start, end):
            if i < len(text):
                char = text[i]
                priority = boundaries.index(char) if char in boundaries else -1
                if priority >= 0 and priority > best_priority:
                    best_pos = i + 1
                    best_priority = priority
        
        return best_pos
    
    def chunk(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Chunk documents adaptively"""
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            chunks = []
            start = 0
            
            while start < len(doc):
                remaining = len(doc) - start
                
                if remaining <= self.max_chunk_size:
                    # Take the rest
                    chunks.append(doc[start:])
                    break
                
                # Find optimal split point
                split_pos = self._find_split_point(
                    doc[start:],
                    self.target_chunk_size
                )
                
                chunk = doc[start:start + split_pos]
                chunks.append(chunk)
                start += split_pos
            
            all_chunks.extend(chunks)
            
            # Add metadata
            base_metadata = metadata_list[idx] if metadata_list else {}
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source_doc_index": idx,
                    "chunk_length": len(chunk),
                    "chunking_strategy": "adaptive"
                }
                all_metadata.append(chunk_meta)
        
        logger.info(f"Created {len(all_chunks)} adaptive chunks from {len(documents)} documents")
        return all_chunks, all_metadata


# Factory function
def get_chunking_strategy(
    strategy: str = "recursive",
    **kwargs
) -> ChunkingStrategy:
    """
    Get a chunking strategy instance
    
    Args:
        strategy: Strategy name (recursive, semantic, fixed, adaptive)
        **kwargs: Strategy-specific parameters
        
    Returns:
        ChunkingStrategy instance
    """
    strategies = {
        "recursive": RecursiveChunking,
        "semantic": SemanticChunking,
        "fixed": FixedSizeChunking,
        "adaptive": AdaptiveChunking
    }
    
    if strategy not in strategies:
        logger.warning(f"Unknown strategy '{strategy}', using 'recursive'")
        strategy = "recursive"
    
    return strategies[strategy](**kwargs)


# Backward compatible function
def chunk_documents(
    docs: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata_list: Optional[List[Dict]] = None
) -> List[str]:
    """
    Chunk documents using recursive strategy (backward compatible)
    
    Args:
        docs: List of documents
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        metadata_list: Optional metadata list
        
    Returns:
        List of chunks
    """
    strategy = RecursiveChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, _ = strategy.chunk(docs, metadata_list)
    return chunks


def chunk_documents_with_metadata(
    docs: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata_list: Optional[List[Dict]] = None,
    strategy: str = "recursive"
) -> Tuple[List[str], List[Dict]]:
    """
    Chunk documents with metadata preservation
    
    Args:
        docs: List of documents
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        metadata_list: Optional metadata list
        strategy: Chunking strategy to use
        
    Returns:
        Tuple of (chunks, chunk_metadata)
    """
    chunker = get_chunking_strategy(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk(docs, metadata_list)
