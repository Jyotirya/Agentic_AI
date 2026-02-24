"""
Verification Pipeline - Integrated verification system
This module integrates citation tracking, conflict detection, and confidence scoring
into a unified verification pipeline for RAG outputs.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import hashlib

from utils.citation_tracker import (
    CitationTracker, Citation, ProvenanceTrace,
    get_citation_tracker, create_citations_from_retrieval
)
from utils.conflict_detector import (
    ConflictDetector, ConflictReport, DetectedConflict,
    get_conflict_detector, detect_conflicts
)
from utils.confidence_scorer import (
    ConfidenceScorer, ConfidenceReport, ConfidenceLevel,
    get_confidence_scorer, calculate_confidence
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Overall verification status"""
    VERIFIED = "verified"               # High confidence, no conflicts
    PARTIALLY_VERIFIED = "partially_verified"  # Moderate confidence or minor conflicts
    NEEDS_REVIEW = "needs_review"       # Low confidence or significant conflicts
    UNVERIFIED = "unverified"           # Cannot verify
    FAILED = "failed"                   # Verification failed


class RiskLevel(Enum):
    """Risk level for using the response"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VerificationSummary:
    """Summary of verification results"""
    status: VerificationStatus
    risk_level: RiskLevel
    confidence_score: float
    confidence_level: str
    conflict_count: int
    conflict_free: bool
    citation_count: int
    source_count: int
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "conflict_count": self.conflict_count,
            "conflict_free": self.conflict_free,
            "citation_count": self.citation_count,
            "source_count": self.source_count,
            "recommendations": self.recommendations
        }


@dataclass
class VerificationResult:
    """Complete verification result"""
    result_id: str
    query: str
    answer: str
    summary: VerificationSummary
    citations: List[Citation]
    provenance_trace: ProvenanceTrace
    conflict_report: ConflictReport
    confidence_report: ConfidenceReport
    verified_at: str
    processing_time_ms: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "result_id": self.result_id,
            "query": self.query,
            "answer_preview": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "summary": self.summary.to_dict(),
            "citation_count": len(self.citations),
            "citations": [c.to_dict() for c in self.citations[:5]],  # Top 5 citations
            "conflict_report_id": self.conflict_report.report_id,
            "confidence_report_id": self.confidence_report.report_id,
            "verified_at": self.verified_at,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }
    
    def get_formatted_citations(self, format_type: str = "detailed") -> str:
        """Get formatted citations"""
        tracker = get_citation_tracker()
        return tracker.format_citations_for_display(self.citations, format_type)
    
    def get_formatted_conflicts(self, format_type: str = "detailed") -> str:
        """Get formatted conflict report"""
        detector = get_conflict_detector()
        return detector.format_conflict_report(self.conflict_report, format_type)
    
    def get_formatted_confidence(self, format_type: str = "detailed") -> str:
        """Get formatted confidence report"""
        scorer = get_confidence_scorer()
        return scorer.format_confidence_report(self.confidence_report, format_type)


class VerificationPipeline:
    """
    Integrated verification pipeline that combines:
    1. Citation provenance tracking
    2. Conflict detection
    3. Confidence scoring
    
    Provides a unified interface for verifying RAG outputs.
    """
    
    def __init__(
        self,
        citation_tracker: Optional[CitationTracker] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None
    ):
        """
        Initialize the verification pipeline.
        
        Args:
            citation_tracker: Optional custom citation tracker
            conflict_detector: Optional custom conflict detector
            confidence_scorer: Optional custom confidence scorer
        """
        self.citation_tracker = citation_tracker or get_citation_tracker()
        self.conflict_detector = conflict_detector or get_conflict_detector()
        self.confidence_scorer = confidence_scorer or get_confidence_scorer()
        
        self.verification_history: List[VerificationResult] = []
        
        # Thresholds for verification status
        self.thresholds = {
            "verified_confidence": 0.75,
            "partial_confidence": 0.50,
            "max_conflicts_verified": 0,
            "max_conflicts_partial": 3,
            "high_risk_confidence": 0.40,
            "critical_risk_confidence": 0.25
        }
    
    def verify(
        self,
        query: str,
        answer: str,
        documents: List[str],
        metadatas: List[Dict],
        retrieval_scores: List[float]
    ) -> VerificationResult:
        """
        Run complete verification pipeline on a RAG response.
        
        Args:
            query: Original user query
            answer: Generated answer
            documents: Retrieved documents
            metadatas: Document metadata
            retrieval_scores: Retrieval relevance scores
            
        Returns:
            Complete VerificationResult
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting verification pipeline for query: {query[:50]}...")
        
        # Step 1: Create citations and provenance trace
        logger.info("Step 1: Creating citations and provenance trace...")
        citations = self.citation_tracker.create_citations_from_retrieval(
            documents, metadatas, retrieval_scores
        )
        
        # Step 2: Detect conflicts
        logger.info("Step 2: Detecting conflicts...")
        conflict_report = self.conflict_detector.detect_conflicts(
            documents, metadatas, query
        )
        
        # Step 3: Calculate confidence
        logger.info("Step 3: Calculating confidence...")
        confidence_report = self.confidence_scorer.calculate_confidence(
            query=query,
            answer=answer,
            documents=documents,
            metadatas=metadatas,
            retrieval_scores=retrieval_scores,
            conflict_report=conflict_report.to_dict()
        )
        
        # Step 4: Create provenance trace
        logger.info("Step 4: Creating provenance trace...")
        provenance_trace = self.citation_tracker.create_provenance_trace(
            query=query,
            answer=answer,
            citations=citations,
            processing_metadata={
                "conflict_report_id": conflict_report.report_id,
                "confidence_report_id": confidence_report.report_id,
                "retrieval_scores": retrieval_scores
            }
        )
        
        # Step 5: Determine verification status and risk
        logger.info("Step 5: Determining verification status...")
        status = self._determine_status(confidence_report, conflict_report)
        risk_level = self._determine_risk_level(confidence_report, conflict_report)
        
        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            status, risk_level, confidence_report, conflict_report
        )
        
        # Create summary
        summary = VerificationSummary(
            status=status,
            risk_level=risk_level,
            confidence_score=confidence_report.breakdown.overall_score,
            confidence_level=confidence_report.breakdown.confidence_level.value,
            conflict_count=len(conflict_report.conflicts),
            conflict_free=conflict_report.conflict_free,
            citation_count=len(citations),
            source_count=len(set(c.source_file for c in citations)),
            recommendations=recommendations
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create final result
        result = VerificationResult(
            result_id=hashlib.md5(
                f"{query}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            query=query,
            answer=answer,
            summary=summary,
            citations=citations,
            provenance_trace=provenance_trace,
            conflict_report=conflict_report,
            confidence_report=confidence_report,
            verified_at=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            metadata={
                "num_documents": len(documents),
                "num_unique_sources": len(set(m.get('source', f's{i}') for i, m in enumerate(metadatas))),
                "query_length": len(query.split()),
                "answer_length": len(answer.split())
            }
        )
        
        # Store in history
        self.verification_history.append(result)
        
        logger.info(
            f"Verification complete: {status.value} "
            f"(confidence: {summary.confidence_score:.1%}, "
            f"conflicts: {summary.conflict_count}, "
            f"time: {processing_time:.1f}ms)"
        )
        
        return result
    
    def _determine_status(
        self,
        confidence_report: ConfidenceReport,
        conflict_report: ConflictReport
    ) -> VerificationStatus:
        """Determine overall verification status"""
        confidence = confidence_report.breakdown.overall_score
        num_conflicts = len(conflict_report.conflicts)
        
        # High confidence and no conflicts = verified
        if confidence >= self.thresholds["verified_confidence"]:
            if num_conflicts <= self.thresholds["max_conflicts_verified"]:
                return VerificationStatus.VERIFIED
            elif num_conflicts <= self.thresholds["max_conflicts_partial"]:
                return VerificationStatus.PARTIALLY_VERIFIED
            else:
                return VerificationStatus.NEEDS_REVIEW
        
        # Moderate confidence
        elif confidence >= self.thresholds["partial_confidence"]:
            if num_conflicts <= self.thresholds["max_conflicts_partial"]:
                return VerificationStatus.PARTIALLY_VERIFIED
            else:
                return VerificationStatus.NEEDS_REVIEW
        
        # Low confidence
        elif confidence >= self.thresholds["critical_risk_confidence"]:
            return VerificationStatus.NEEDS_REVIEW
        
        # Very low confidence
        else:
            return VerificationStatus.UNVERIFIED
    
    def _determine_risk_level(
        self,
        confidence_report: ConfidenceReport,
        conflict_report: ConflictReport
    ) -> RiskLevel:
        """Determine risk level for using the response"""
        confidence = confidence_report.breakdown.overall_score
        reliability_impact = conflict_report.overall_reliability_impact
        
        # Critical risk: very low confidence or high conflict impact
        if confidence < self.thresholds["critical_risk_confidence"]:
            return RiskLevel.CRITICAL
        if reliability_impact > 0.5:
            return RiskLevel.CRITICAL
        
        # High risk
        if confidence < self.thresholds["high_risk_confidence"]:
            return RiskLevel.HIGH
        if reliability_impact > 0.3:
            return RiskLevel.HIGH
        
        # Moderate risk
        if confidence < self.thresholds["partial_confidence"]:
            return RiskLevel.MODERATE
        if reliability_impact > 0.1:
            return RiskLevel.MODERATE
        
        # Low risk
        return RiskLevel.LOW
    
    def _generate_recommendations(
        self,
        status: VerificationStatus,
        risk_level: RiskLevel,
        confidence_report: ConfidenceReport,
        conflict_report: ConflictReport
    ) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        # Status-based recommendations
        if status == VerificationStatus.VERIFIED:
            recommendations.append(
                "✅ Response is well-verified and can be used with confidence"
            )
        elif status == VerificationStatus.PARTIALLY_VERIFIED:
            recommendations.append(
                "⚠️ Response is partially verified - review specific details before use"
            )
        elif status == VerificationStatus.NEEDS_REVIEW:
            recommendations.append(
                "🔍 Response needs manual review - verify critical information from primary sources"
            )
        elif status == VerificationStatus.UNVERIFIED:
            recommendations.append(
                "❌ Response could not be adequately verified - use with caution"
            )
        
        # Risk-based recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(
                "🚨 CRITICAL: Do not use this response for important decisions without verification"
            )
        elif risk_level == RiskLevel.HIGH:
            recommendations.append(
                "⚠️ HIGH RISK: Verify key facts before relying on this information"
            )
        
        # Add conflict-specific recommendations
        recommendations.extend(conflict_report.recommendations[:2])
        
        # Add confidence-specific recommendations
        recommendations.extend(confidence_report.recommendations[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)
        
        return unique_recommendations[:5]  # Limit to 5 recommendations
    
    def quick_verify(
        self,
        documents: List[str],
        metadatas: List[Dict],
        retrieval_scores: List[float]
    ) -> Tuple[float, bool, int]:
        """
        Quick verification without full report generation.
        
        Args:
            documents: Retrieved documents
            metadatas: Document metadata
            retrieval_scores: Retrieval scores
            
        Returns:
            Tuple of (confidence_score, conflict_free, citation_count)
        """
        # Quick confidence estimate
        num_sources = len(set(m.get('source', f's{i}') for i, m in enumerate(metadatas)))
        confidence = self.confidence_scorer.get_quick_confidence(
            retrieval_scores, num_sources
        )
        
        # Quick conflict check (only if needed)
        conflict_free = True
        if len(documents) > 1:
            # Sample check for obvious conflicts
            conflict_report = self.conflict_detector.detect_conflicts(
                documents[:3], metadatas[:3]  # Check first 3 only
            )
            conflict_free = conflict_report.conflict_free
        
        return confidence, conflict_free, len(documents)
    
    def format_verification_result(
        self,
        result: VerificationResult,
        format_type: str = "full",
        include_citations: bool = True,
        include_conflicts: bool = True,
        include_confidence: bool = True
    ) -> str:
        """
        Format verification result for display.
        
        Args:
            result: Verification result to format
            format_type: "full", "summary", or "compact"
            include_citations: Whether to include citation details
            include_conflicts: Whether to include conflict details
            include_confidence: Whether to include confidence details
            
        Returns:
            Formatted string
        """
        if format_type == "compact":
            return self._format_compact(result)
        elif format_type == "summary":
            return self._format_summary(result)
        else:
            return self._format_full(
                result, include_citations, include_conflicts, include_confidence
            )
    
    def _format_compact(self, result: VerificationResult) -> str:
        """Format compact verification result"""
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️",
            VerificationStatus.NEEDS_REVIEW: "🔍",
            VerificationStatus.UNVERIFIED: "❌",
            VerificationStatus.FAILED: "💥"
        }
        
        risk_emoji = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MODERATE: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴"
        }
        
        s = result.summary
        return (
            f"{status_emoji.get(s.status, '❓')} {s.status.value.replace('_', ' ').title()} | "
            f"Confidence: {s.confidence_score:.0%} | "
            f"Risk: {risk_emoji.get(s.risk_level, '⚪')} {s.risk_level.value.title()} | "
            f"Sources: {s.source_count}"
        )
    
    def _format_summary(self, result: VerificationResult) -> str:
        """Format summary verification result"""
        s = result.summary
        
        lines = [
            "═" * 50,
            "📋 VERIFICATION SUMMARY",
            "═" * 50,
            f"Status: {s.status.value.replace('_', ' ').title()}",
            f"Risk Level: {s.risk_level.value.title()}",
            f"Confidence: {s.confidence_score:.1%} ({s.confidence_level})",
            f"Conflicts: {s.conflict_count} ({'None' if s.conflict_free else 'Detected'})",
            f"Citations: {s.citation_count} from {s.source_count} source(s)",
            "",
            "Recommendations:",
        ]
        
        for rec in s.recommendations[:3]:
            lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def _format_full(
        self,
        result: VerificationResult,
        include_citations: bool,
        include_conflicts: bool,
        include_confidence: bool
    ) -> str:
        """Format full verification result"""
        lines = [
            "═" * 70,
            "🔍 COMPLETE VERIFICATION REPORT",
            "═" * 70,
            f"Report ID: {result.result_id}",
            f"Verified At: {result.verified_at}",
            f"Processing Time: {result.processing_time_ms:.1f}ms",
            "",
        ]
        
        # Summary section
        lines.append(self._format_summary(result))
        lines.append("")
        
        # Citations section
        if include_citations and result.citations:
            lines.append("─" * 70)
            lines.append(result.get_formatted_citations("compact"))
            lines.append("")
        
        # Conflicts section
        if include_conflicts:
            lines.append("─" * 70)
            lines.append(result.get_formatted_conflicts("compact"))
            lines.append("")
        
        # Confidence section
        if include_confidence:
            lines.append("─" * 70)
            lines.append(result.get_formatted_confidence("summary"))
            lines.append("")
        
        lines.append("═" * 70)
        
        return "\n".join(lines)
    
    def get_verification_history(self, limit: int = 10) -> List[VerificationResult]:
        """Get recent verification results"""
        return self.verification_history[-limit:]
    
    def export_result(self, result: VerificationResult) -> str:
        """Export verification result as JSON"""
        return json.dumps(result.to_dict(), indent=2)
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics from verification history"""
        if not self.verification_history:
            return {"total_verifications": 0}
        
        total = len(self.verification_history)
        
        # Status distribution
        status_counts = {}
        for result in self.verification_history:
            status = result.summary.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Risk distribution
        risk_counts = {}
        for result in self.verification_history:
            risk = result.summary.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Average confidence
        avg_confidence = sum(
            r.summary.confidence_score for r in self.verification_history
        ) / total
        
        # Average processing time
        avg_time = sum(
            r.processing_time_ms for r in self.verification_history
        ) / total
        
        return {
            "total_verifications": total,
            "status_distribution": status_counts,
            "risk_distribution": risk_counts,
            "average_confidence": avg_confidence,
            "average_processing_time_ms": avg_time,
            "verified_rate": status_counts.get("verified", 0) / total,
            "conflict_free_rate": sum(
                1 for r in self.verification_history if r.summary.conflict_free
            ) / total
        }


# Global instance
_verification_pipeline = None

def get_verification_pipeline() -> VerificationPipeline:
    """Get or create verification pipeline instance"""
    global _verification_pipeline
    if _verification_pipeline is None:
        _verification_pipeline = VerificationPipeline()
    return _verification_pipeline


def verify_response(
    query: str,
    answer: str,
    documents: List[str],
    metadatas: List[Dict],
    retrieval_scores: List[float]
) -> VerificationResult:
    """
    Convenience function to verify a RAG response.
    
    Args:
        query: Original query
        answer: Generated answer
        documents: Retrieved documents
        metadatas: Document metadata
        retrieval_scores: Retrieval scores
        
    Returns:
        VerificationResult
    """
    pipeline = get_verification_pipeline()
    return pipeline.verify(
        query, answer, documents, metadatas, retrieval_scores
    )


def quick_verify(
    documents: List[str],
    metadatas: List[Dict],
    retrieval_scores: List[float]
) -> Tuple[float, bool, int]:
    """
    Convenience function for quick verification.
    
    Args:
        documents: Retrieved documents
        metadatas: Document metadata
        retrieval_scores: Retrieval scores
        
    Returns:
        Tuple of (confidence, conflict_free, citation_count)
    """
    pipeline = get_verification_pipeline()
    return pipeline.quick_verify(documents, metadatas, retrieval_scores)
