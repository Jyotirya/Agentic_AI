"""
Citation Provenance Tracking Module
This module provides comprehensive citation tracking for verifiable RAG outputs.
It tracks the source, location, and relevance of each piece of information.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import re
from collections import defaultdict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of information sources"""
    COURSE = "course"
    FACULTY = "faculty"
    ANNOUNCEMENT = "announcement"
    RESEARCH = "research"
    GENERAL = "general"
    UNKNOWN = "unknown"


class CitationReliability(Enum):
    """Reliability levels for citations"""
    HIGH = "high"           # Official documents, verified sources
    MEDIUM = "medium"       # Semi-official, recent updates
    LOW = "low"             # Outdated, unverified
    UNCERTAIN = "uncertain" # Cannot determine reliability


@dataclass
class Citation:
    """Represents a single citation with provenance information"""
    citation_id: str
    source_file: str
    source_type: SourceType
    document_id: str
    chunk_index: int
    content_snippet: str
    full_content: str
    relevance_score: float
    timestamp: str
    metadata: Dict[str, Any]
    reliability: CitationReliability = CitationReliability.MEDIUM
    extraction_method: str = "semantic_retrieval"
    
    # Provenance tracking
    original_position: Optional[int] = None
    content_hash: Optional[str] = None
    parent_document_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate content hash after initialization"""
        if self.content_hash is None:
            self.content_hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate a unique hash for the content"""
        content_to_hash = f"{self.source_file}:{self.document_id}:{self.full_content}"
        return hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['source_type'] = self.source_type.value
        result['reliability'] = self.reliability.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Citation':
        """Create from dictionary"""
        data['source_type'] = SourceType(data['source_type'])
        data['reliability'] = CitationReliability(data['reliability'])
        return cls(**data)


@dataclass
class CitationCluster:
    """A cluster of related citations supporting the same claim"""
    cluster_id: str
    citations: List[Citation]
    claim_text: str
    aggregate_score: float
    agreement_level: float  # How well citations agree (0-1)
    source_diversity: int   # Number of unique sources
    
    def add_citation(self, citation: Citation):
        """Add a citation to the cluster"""
        self.citations.append(citation)
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate aggregate metrics"""
        if not self.citations:
            self.aggregate_score = 0.0
            self.agreement_level = 0.0
            self.source_diversity = 0
            return
        
        self.aggregate_score = sum(c.relevance_score for c in self.citations) / len(self.citations)
        unique_sources = set(c.source_file for c in self.citations)
        self.source_diversity = len(unique_sources)


@dataclass
class ProvenanceTrace:
    """Complete provenance trace for a generated answer"""
    trace_id: str
    query: str
    generated_answer: str
    citations: List[Citation]
    citation_clusters: List[CitationCluster]
    timestamp: str
    processing_metadata: Dict[str, Any]
    
    # Quality metrics
    overall_confidence: float = 0.0
    source_coverage: float = 0.0  # How much of answer is supported
    citation_density: float = 0.0  # Citations per sentence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "generated_answer": self.generated_answer,
            "citations": [c.to_dict() for c in self.citations],
            "citation_clusters": [
                {
                    "cluster_id": cc.cluster_id,
                    "claim_text": cc.claim_text,
                    "aggregate_score": cc.aggregate_score,
                    "agreement_level": cc.agreement_level,
                    "source_diversity": cc.source_diversity,
                    "citation_ids": [c.citation_id for c in cc.citations]
                }
                for cc in self.citation_clusters
            ],
            "timestamp": self.timestamp,
            "processing_metadata": self.processing_metadata,
            "overall_confidence": self.overall_confidence,
            "source_coverage": self.source_coverage,
            "citation_density": self.citation_density
        }


class CitationTracker:
    """
    Comprehensive citation tracking system for RAG outputs.
    Provides provenance tracking, citation clustering, and verification.
    """
    
    def __init__(self):
        """Initialize the citation tracker"""
        self.citation_history: List[ProvenanceTrace] = []
        self.citation_index: Dict[str, Citation] = {}
        self.source_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_citations": 0,
            "avg_relevance": 0.0,
            "last_used": None
        })
    
    def create_citation(
        self,
        content: str,
        metadata: Dict[str, Any],
        relevance_score: float,
        chunk_index: int = 0
    ) -> Citation:
        """
        Create a citation from retrieved content and metadata.
        
        Args:
            content: The retrieved text content
            metadata: Metadata from the vector database
            relevance_score: Relevance score from retrieval
            chunk_index: Index of the chunk in retrieval results
            
        Returns:
            Citation object with full provenance information
        """
        citation_id = str(uuid.uuid4())[:8]
        
        # Determine source type
        source_type_str = metadata.get('source_type', 'unknown')
        try:
            source_type = SourceType(source_type_str)
        except ValueError:
            source_type = SourceType.UNKNOWN
        
        # Determine reliability based on metadata
        reliability = self._assess_reliability(metadata, source_type)
        
        # Create snippet (first 200 chars)
        snippet = content[:200] + "..." if len(content) > 200 else content
        
        citation = Citation(
            citation_id=citation_id,
            source_file=metadata.get('source', 'unknown'),
            source_type=source_type,
            document_id=metadata.get('doc_id', f'doc_{chunk_index}'),
            chunk_index=chunk_index,
            content_snippet=snippet,
            full_content=content,
            relevance_score=relevance_score,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
            reliability=reliability,
            original_position=metadata.get('chunk_index'),
            parent_document_id=metadata.get('source_doc_index')
        )
        
        # Index the citation
        self.citation_index[citation_id] = citation
        
        # Update source stats
        source = metadata.get('source', 'unknown')
        self.source_stats[source]['total_citations'] += 1
        self.source_stats[source]['last_used'] = datetime.now().isoformat()
        
        return citation
    
    def create_citations_from_retrieval(
        self,
        documents: List[str],
        metadatas: List[Dict],
        scores: List[float]
    ) -> List[Citation]:
        """
        Create citations from retrieval results.
        
        Args:
            documents: Retrieved document texts
            metadatas: Document metadata list
            scores: Relevance scores
            
        Returns:
            List of Citation objects
        """
        citations = []
        for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
            citation = self.create_citation(
                content=doc,
                metadata=meta,
                relevance_score=score,
                chunk_index=i
            )
            citations.append(citation)
        
        logger.info(f"Created {len(citations)} citations from retrieval results")
        return citations
    
    def _assess_reliability(
        self,
        metadata: Dict[str, Any],
        source_type: SourceType
    ) -> CitationReliability:
        """
        Assess the reliability of a citation based on metadata.
        
        Args:
            metadata: Citation metadata
            source_type: Type of source
            
        Returns:
            CitationReliability level
        """
        # Official sources are more reliable
        if source_type in [SourceType.COURSE, SourceType.FACULTY]:
            return CitationReliability.HIGH
        
        # Check for timestamp/freshness
        timestamp_str = metadata.get('timestamp', '')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - timestamp).days
                if age_days > 365:
                    return CitationReliability.LOW
            except (ValueError, TypeError):
                pass
        
        # Announcements are time-sensitive
        if source_type == SourceType.ANNOUNCEMENT:
            return CitationReliability.MEDIUM
        
        return CitationReliability.MEDIUM
    
    def cluster_citations(
        self,
        citations: List[Citation],
        similarity_threshold: float = 0.7
    ) -> List[CitationCluster]:
        """
        Cluster citations that support similar claims.
        
        Args:
            citations: List of citations to cluster
            similarity_threshold: Threshold for clustering
            
        Returns:
            List of CitationClusters
        """
        if not citations:
            return []
        
        clusters = []
        used_citations: Set[str] = set()
        
        for citation in citations:
            if citation.citation_id in used_citations:
                continue
            
            # Find similar citations
            cluster_citations = [citation]
            used_citations.add(citation.citation_id)
            
            for other in citations:
                if other.citation_id in used_citations:
                    continue
                
                # Check content similarity
                similarity = self._compute_content_similarity(
                    citation.full_content,
                    other.full_content
                )
                
                if similarity >= similarity_threshold:
                    cluster_citations.append(other)
                    used_citations.add(other.citation_id)
            
            # Create cluster
            cluster = CitationCluster(
                cluster_id=str(uuid.uuid4())[:8],
                citations=cluster_citations,
                claim_text=citation.content_snippet,
                aggregate_score=sum(c.relevance_score for c in cluster_citations) / len(cluster_citations),
                agreement_level=self._compute_agreement_level(cluster_citations),
                source_diversity=len(set(c.source_file for c in cluster_citations))
            )
            clusters.append(cluster)
        
        logger.info(f"Created {len(clusters)} citation clusters from {len(citations)} citations")
        return clusters
    
    def _compute_content_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two text contents using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_agreement_level(self, citations: List[Citation]) -> float:
        """
        Compute how well citations agree with each other.
        
        Args:
            citations: List of citations in a cluster
            
        Returns:
            Agreement level between 0 and 1
        """
        if len(citations) <= 1:
            return 1.0
        
        # Compute average pairwise similarity
        total_similarity = 0.0
        pair_count = 0
        
        for i, c1 in enumerate(citations):
            for c2 in citations[i + 1:]:
                similarity = self._compute_content_similarity(
                    c1.full_content,
                    c2.full_content
                )
                total_similarity += similarity
                pair_count += 1
        
        return total_similarity / pair_count if pair_count > 0 else 1.0
    
    def create_provenance_trace(
        self,
        query: str,
        answer: str,
        citations: List[Citation],
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceTrace:
        """
        Create a complete provenance trace for a generated answer.
        
        Args:
            query: Original user query
            answer: Generated answer
            citations: Citations used for the answer
            processing_metadata: Additional processing information
            
        Returns:
            Complete ProvenanceTrace object
        """
        # Cluster citations
        clusters = self.cluster_citations(citations)
        
        # Calculate metrics
        source_coverage = self._calculate_source_coverage(answer, citations)
        citation_density = self._calculate_citation_density(answer, citations)
        overall_confidence = self._calculate_overall_confidence(citations, clusters)
        
        trace = ProvenanceTrace(
            trace_id=str(uuid.uuid4())[:12],
            query=query,
            generated_answer=answer,
            citations=citations,
            citation_clusters=clusters,
            timestamp=datetime.now().isoformat(),
            processing_metadata=processing_metadata or {},
            overall_confidence=overall_confidence,
            source_coverage=source_coverage,
            citation_density=citation_density
        )
        
        # Store in history
        self.citation_history.append(trace)
        
        logger.info(f"Created provenance trace {trace.trace_id} with {len(citations)} citations")
        return trace
    
    def _calculate_source_coverage(
        self,
        answer: str,
        citations: List[Citation]
    ) -> float:
        """
        Calculate how much of the answer is supported by citations.
        
        Args:
            answer: Generated answer
            citations: Supporting citations
            
        Returns:
            Coverage score between 0 and 1
        """
        if not answer or not citations:
            return 0.0
        
        answer_words = set(answer.lower().split())
        citation_words = set()
        
        for citation in citations:
            citation_words.update(citation.full_content.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                       'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                       'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                       'through', 'during', 'before', 'after', 'above', 'below',
                       'between', 'under', 'again', 'further', 'then', 'once'}
        
        answer_words -= common_words
        
        if not answer_words:
            return 0.5  # Default for short answers
        
        supported_words = answer_words & citation_words
        return len(supported_words) / len(answer_words)
    
    def _calculate_citation_density(
        self,
        answer: str,
        citations: List[Citation]
    ) -> float:
        """
        Calculate citation density (citations per sentence).
        
        Args:
            answer: Generated answer
            citations: Supporting citations
            
        Returns:
            Citation density score
        """
        # Count sentences
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count == 0:
            return 0.0
        
        return len(citations) / sentence_count
    
    def _calculate_overall_confidence(
        self,
        citations: List[Citation],
        clusters: List[CitationCluster]
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            citations: All citations
            clusters: Citation clusters
            
        Returns:
            Confidence score between 0 and 1
        """
        if not citations:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Average relevance score
        avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
        
        # 2. Source diversity
        unique_sources = len(set(c.source_file for c in citations))
        diversity_bonus = min(unique_sources * 0.1, 0.3)
        
        # 3. Reliability
        reliability_scores = {
            CitationReliability.HIGH: 1.0,
            CitationReliability.MEDIUM: 0.7,
            CitationReliability.LOW: 0.4,
            CitationReliability.UNCERTAIN: 0.3
        }
        avg_reliability = sum(
            reliability_scores[c.reliability] for c in citations
        ) / len(citations)
        
        # 4. Agreement level from clusters
        avg_agreement = 0.0
        if clusters:
            avg_agreement = sum(c.agreement_level for c in clusters) / len(clusters)
        
        # Weighted combination
        confidence = (
            avg_relevance * 0.35 +
            avg_reliability * 0.25 +
            avg_agreement * 0.25 +
            diversity_bonus * 0.15
        )
        
        return min(confidence, 1.0)
    
    def format_citations_for_display(
        self,
        citations: List[Citation],
        format_type: str = "detailed"
    ) -> str:
        """
        Format citations for human-readable display.
        
        Args:
            citations: List of citations
            format_type: "detailed", "compact", or "inline"
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return "No citations available."
        
        if format_type == "inline":
            return self._format_inline_citations(citations)
        elif format_type == "compact":
            return self._format_compact_citations(citations)
        else:
            return self._format_detailed_citations(citations)
    
    def _format_detailed_citations(self, citations: List[Citation]) -> str:
        """Format citations in detailed format"""
        lines = ["📚 **Sources and Citations:**\n"]
        
        for i, citation in enumerate(citations, 1):
            lines.append(f"**[{i}] {citation.source_type.value.title()} Source**")
            lines.append(f"  - File: `{citation.source_file}`")
            lines.append(f"  - Document ID: `{citation.document_id}`")
            lines.append(f"  - Relevance: {citation.relevance_score:.2%}")
            lines.append(f"  - Reliability: {citation.reliability.value.title()}")
            lines.append(f"  - Content: \"{citation.content_snippet}\"")
            lines.append(f"  - Hash: `{citation.content_hash}`")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_compact_citations(self, citations: List[Citation]) -> str:
        """Format citations in compact format"""
        lines = ["**Sources:**"]
        
        for i, citation in enumerate(citations, 1):
            lines.append(
                f"[{i}] {citation.source_file} ({citation.source_type.value}) "
                f"- {citation.relevance_score:.0%} relevance"
            )
        
        return "\n".join(lines)
    
    def _format_inline_citations(self, citations: List[Citation]) -> str:
        """Format citations for inline display"""
        sources = set(c.source_file for c in citations)
        return f"Sources: {', '.join(sources)}"
    
    def get_citation_by_id(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by its ID"""
        return self.citation_index.get(citation_id)
    
    def get_provenance_history(
        self,
        limit: int = 10
    ) -> List[ProvenanceTrace]:
        """Get recent provenance traces"""
        return self.citation_history[-limit:]
    
    def export_provenance_trace(
        self,
        trace: ProvenanceTrace
    ) -> str:
        """Export provenance trace as JSON"""
        return json.dumps(trace.to_dict(), indent=2)
    
    def verify_citation(self, citation: Citation) -> Dict[str, Any]:
        """
        Verify a citation's integrity.
        
        Args:
            citation: Citation to verify
            
        Returns:
            Verification result
        """
        current_hash = hashlib.sha256(
            f"{citation.source_file}:{citation.document_id}:{citation.full_content}".encode()
        ).hexdigest()[:16]
        
        is_valid = current_hash == citation.content_hash
        
        return {
            "citation_id": citation.citation_id,
            "is_valid": is_valid,
            "original_hash": citation.content_hash,
            "current_hash": current_hash,
            "verification_time": datetime.now().isoformat()
        }


# Global instance
_citation_tracker = None

def get_citation_tracker() -> CitationTracker:
    """Get or create citation tracker instance"""
    global _citation_tracker
    if _citation_tracker is None:
        _citation_tracker = CitationTracker()
    return _citation_tracker


def create_citations_from_retrieval(
    documents: List[str],
    metadatas: List[Dict],
    scores: List[float]
) -> List[Citation]:
    """
    Convenience function to create citations from retrieval results.
    
    Args:
        documents: Retrieved documents
        metadatas: Document metadata
        scores: Relevance scores
        
    Returns:
        List of Citation objects
    """
    tracker = get_citation_tracker()
    return tracker.create_citations_from_retrieval(documents, metadatas, scores)
