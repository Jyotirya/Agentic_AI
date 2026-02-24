"""
Utils Package - Utility modules for the RAG system
Includes citation tracking, conflict detection, confidence scoring, and verification pipeline.
"""

from utils.citation_tracker import (
    CitationTracker,
    Citation,
    ProvenanceTrace,
    CitationCluster,
    SourceType,
    CitationReliability,
    get_citation_tracker,
    create_citations_from_retrieval
)

from utils.conflict_detector import (
    ConflictDetector,
    DetectedConflict,
    ConflictReport,
    ConflictEvidence,
    ConflictType,
    ConflictSeverity,
    NumericalExtractor,
    TemporalExtractor,
    EntityExtractor,
    get_conflict_detector,
    detect_conflicts
)

from utils.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceReport,
    ConfidenceBreakdown,
    ComponentScore,
    ConfidenceLevel,
    ConfidenceComponent,
    get_confidence_scorer,
    calculate_confidence,
    get_quick_confidence
)

from utils.verification_pipeline import (
    VerificationPipeline,
    VerificationResult,
    VerificationSummary,
    VerificationStatus,
    RiskLevel,
    get_verification_pipeline,
    verify_response,
    quick_verify
)

__all__ = [
    # Citation Tracking
    'CitationTracker',
    'Citation',
    'ProvenanceTrace',
    'CitationCluster',
    'SourceType',
    'CitationReliability',
    'get_citation_tracker',
    'create_citations_from_retrieval',
    
    # Conflict Detection
    'ConflictDetector',
    'DetectedConflict',
    'ConflictReport',
    'ConflictEvidence',
    'ConflictType',
    'ConflictSeverity',
    'NumericalExtractor',
    'TemporalExtractor',
    'EntityExtractor',
    'get_conflict_detector',
    'detect_conflicts',
    
    # Confidence Scoring
    'ConfidenceScorer',
    'ConfidenceReport',
    'ConfidenceBreakdown',
    'ComponentScore',
    'ConfidenceLevel',
    'ConfidenceComponent',
    'get_confidence_scorer',
    'calculate_confidence',
    'get_quick_confidence',
    
    # Verification Pipeline
    'VerificationPipeline',
    'VerificationResult',
    'VerificationSummary',
    'VerificationStatus',
    'RiskLevel',
    'get_verification_pipeline',
    'verify_response',
    'quick_verify'
]
