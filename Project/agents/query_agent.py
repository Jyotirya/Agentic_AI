"""
Query Agent - Handles semantic search and context retrieval
This module manages retrieving relevant context from the vector database with reranking.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class QueryManager:
    """Manages query processing and context retrieval"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_reranker: bool = True
    ):
        """
        Initialize the query manager
        
        Args:
            model_name: Name of the sentence transformer model
            use_reranker: Whether to use cross-encoder reranking
        """
        logger.info(f"Initializing QueryManager with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        
        if use_reranker:
            logger.info(f"Loading reranker model: {RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None
    
    def retrieve_context(
        self,
        collection: chromadb.Collection,
        query: str,
        k: int = 4,
        rerank_top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve relevant context with optional reranking
        
        Args:
            collection: ChromaDB collection
            query: User query
            k: Number of final results to return
            rerank_top_k: Number of candidates for reranking (if enabled)
            filter_metadata: Optional metadata filter
            
        Returns:
            Tuple of (documents, metadata, scores)
        """
        try:
            # Embed query
            q_emb = self.model.encode([query]).tolist()
            
            # Retrieve more candidates if reranking is enabled
            n_results = rerank_top_k if self.use_reranker else k
            
            # Query ChromaDB
            query_kwargs = {
                "query_embeddings": q_emb,
                "n_results": min(n_results, collection.count())
            }
            
            if filter_metadata:
                query_kwargs["where"] = filter_metadata
            
            results = collection.query(**query_kwargs)
            
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Rerank if enabled
            if self.use_reranker and len(documents) > 0:
                documents, metadatas, scores = self._rerank_results(
                    query, documents, metadatas, k
                )
            else:
                # Convert distances to similarity scores (1 - distance for cosine)
                scores = [1 - d for d in distances[:k]]
                documents = documents[:k]
                metadatas = metadatas[:k]
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], [], []
    
    def _rerank_results(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict],
        top_k: int
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Rerank retrieved documents using cross-encoder
        
        Args:
            query: User query
            documents: Retrieved documents
            metadatas: Document metadata
            top_k: Number of top results to return
            
        Returns:
            Tuple of (reranked_documents, reranked_metadata, scores)
        """
        if not documents:
            return [], [], []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_meta = [metadatas[i] for i in sorted_indices]
        reranked_scores = [float(scores[i]) for i in sorted_indices]
        
        logger.info(f"Reranked {len(documents)} documents to top {len(reranked_docs)}")
        return reranked_docs, reranked_meta, reranked_scores
    
    def retrieve_with_filters(
        self,
        collection: chromadb.Collection,
        query: str,
        source_type: Optional[str] = None,
        k: int = 4
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve context with metadata filters
        
        Args:
            collection: ChromaDB collection
            query: User query
            source_type: Optional filter by source type (course, faculty, etc.)
            k: Number of results
            
        Returns:
            Tuple of (documents, metadata, scores)
        """
        filter_dict = None
        if source_type:
            filter_dict = {"source_type": source_type}
        
        return self.retrieve_context(
            collection=collection,
            query=query,
            k=k,
            filter_metadata=filter_dict
        )
    
    def hybrid_search(
        self,
        collection: chromadb.Collection,
        query: str,
        k: int = 4,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            collection: ChromaDB collection
            query: User query
            k: Number of results
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
            
        Returns:
            Tuple of (documents, metadata, scores)
        """
        # Get semantic results
        semantic_docs, semantic_meta, semantic_scores = self.retrieve_context(
            collection, query, k=k * 2, rerank_top_k=k * 3
        )
        
        # Simple keyword matching (can be enhanced with BM25)
        query_terms = set(query.lower().split())
        keyword_scores = []
        
        for doc in semantic_docs:
            doc_terms = set(doc.lower().split())
            # Jaccard similarity
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            keyword_score = intersection / union if union > 0 else 0
            keyword_scores.append(keyword_score)
        
        # Normalize scores
        if semantic_scores:
            max_semantic = max(semantic_scores) if max(semantic_scores) > 0 else 1
            semantic_scores = [s / max_semantic for s in semantic_scores]
        
        if keyword_scores:
            max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1
            keyword_scores = [s / max_keyword for s in keyword_scores]
        
        # Combine scores
        combined_scores = [
            semantic_weight * sem + keyword_weight * key
            for sem, key in zip(semantic_scores, keyword_scores)
        ]
        
        # Sort and return top k
        sorted_indices = np.argsort(combined_scores)[::-1][:k]
        
        final_docs = [semantic_docs[i] for i in sorted_indices]
        final_meta = [semantic_meta[i] for i in sorted_indices]
        final_scores = [combined_scores[i] for i in sorted_indices]
        
        logger.info(f"Hybrid search returned {len(final_docs)} documents")
        return final_docs, final_meta, final_scores
    
    def get_query_stats(self, query: str) -> Dict[str, any]:
        """
        Get statistics about a query
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query statistics
        """
        embedding = self.model.encode([query])[0]
        
        return {
            "query": query,
            "query_length": len(query),
            "word_count": len(query.split()),
            "embedding_dimension": len(embedding),
            "embedding_norm": float(np.linalg.norm(embedding))
        }


# Global query manager
_query_manager = None

def get_query_manager() -> QueryManager:
    """Get or create query manager instance"""
    global _query_manager
    if _query_manager is None:
        _query_manager = QueryManager()
    return _query_manager


def retrieve_context(
    collection: chromadb.Collection,
    query: str,
    k: int = 4
) -> List[str]:
    """
    Retrieve context from collection (backward compatible)
    
    Args:
        collection: ChromaDB collection
        query: User query
        k: Number of results
        
    Returns:
        List of document strings
    """
    manager = get_query_manager()
    documents, _, _ = manager.retrieve_context(collection, query, k)
    return documents


def retrieve_context_with_metadata(
    collection: chromadb.Collection,
    query: str,
    k: int = 4
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Retrieve context with metadata and scores
    
    Args:
        collection: ChromaDB collection
        query: User query
        k: Number of results
        
    Returns:
        Tuple of (documents, metadata, scores)
    """
    manager = get_query_manager()
    return manager.retrieve_context(collection, query, k)
