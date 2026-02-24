"""
Test Suite for Verification Components
Tests for citation tracking, conflict detection, and confidence scoring.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.citation_tracker import (
    CitationTracker, Citation, ProvenanceTrace, CitationCluster,
    SourceType, CitationReliability, get_citation_tracker,
    create_citations_from_retrieval
)
from utils.conflict_detector import (
    ConflictDetector, DetectedConflict, ConflictReport,
    ConflictType, ConflictSeverity, NumericalExtractor,
    TemporalExtractor, EntityExtractor, get_conflict_detector,
    detect_conflicts
)
from utils.confidence_scorer import (
    ConfidenceScorer, ConfidenceReport, ConfidenceLevel,
    ConfidenceBreakdown, ComponentScore, ConfidenceComponent,
    get_confidence_scorer, calculate_confidence
)
from utils.verification_pipeline import (
    VerificationPipeline, VerificationResult, VerificationStatus,
    RiskLevel, VerificationSummary, get_verification_pipeline,
    verify_response, quick_verify
)


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Course Code: EE720\nCourse Title: Advanced Power Electronics\nInstructor: Dr. Sharma\nCredits: 6\nPrerequisites: EE301, EE302",
        "Course Code: EE720\nDescription: This course covers advanced topics in power electronics including converter design and control systems.",
        "Name: Dr. R.K. Sharma\nPosition: Professor\nDepartment: Electrical Engineering\nResearch Interests: Power Electronics, Renewable Energy\nEmail: rksharma@iitb.ac.in",
        "Title: PhD Admissions Open\nDate: 2024-01-15\nDeadline: 2024-02-28\nContent: Applications are invited for PhD positions in the EE department."
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for testing"""
    return [
        {"source": "courses.json", "source_type": "course", "doc_id": "courses_0", "timestamp": datetime.now().isoformat()},
        {"source": "courses.json", "source_type": "course", "doc_id": "courses_1", "timestamp": datetime.now().isoformat()},
        {"source": "faculty.json", "source_type": "faculty", "doc_id": "faculty_0", "timestamp": datetime.now().isoformat()},
        {"source": "announcements.json", "source_type": "announcement", "doc_id": "announcements_0", "timestamp": datetime.now().isoformat()}
    ]


@pytest.fixture
def sample_scores():
    """Sample retrieval scores"""
    return [0.85, 0.78, 0.72, 0.65]


@pytest.fixture
def conflicting_documents():
    """Documents with conflicts for testing"""
    return [
        "Course: EE720\nCredits: 6\nInstructor: Dr. Sharma\nPrerequisites: EE301",
        "Course: EE720\nCredits: 8\nInstructor: Dr. Kumar\nPrerequisites: EE302",  # Conflicts!
        "Course: EE720\nDescription: Power electronics course offered in autumn semester."
    ]


@pytest.fixture
def conflicting_metadatas():
    """Metadata for conflicting documents"""
    return [
        {"source": "courses_2024.json", "source_type": "course", "doc_id": "c1", "timestamp": "2024-01-01T00:00:00"},
        {"source": "courses_2023.json", "source_type": "course", "doc_id": "c2", "timestamp": "2023-01-01T00:00:00"},
        {"source": "catalog.json", "source_type": "course", "doc_id": "c3", "timestamp": "2024-06-01T00:00:00"}
    ]


# ============================================================
# Citation Tracker Tests
# ============================================================

class TestCitationTracker:
    """Tests for CitationTracker"""
    
    def test_create_citation(self, sample_documents, sample_metadatas, sample_scores):
        """Test creating a single citation"""
        tracker = CitationTracker()
        
        citation = tracker.create_citation(
            content=sample_documents[0],
            metadata=sample_metadatas[0],
            relevance_score=sample_scores[0],
            chunk_index=0
        )
        
        assert citation is not None
        assert citation.citation_id is not None
        assert citation.source_file == "courses.json"
        assert citation.source_type == SourceType.COURSE
        assert citation.relevance_score == 0.85
        assert citation.content_hash is not None
    
    def test_create_citations_from_retrieval(self, sample_documents, sample_metadatas, sample_scores):
        """Test creating citations from retrieval results"""
        tracker = CitationTracker()
        
        citations = tracker.create_citations_from_retrieval(
            documents=sample_documents,
            metadatas=sample_metadatas,
            scores=sample_scores
        )
        
        assert len(citations) == 4
        assert all(isinstance(c, Citation) for c in citations)
        assert citations[0].source_type == SourceType.COURSE
        assert citations[2].source_type == SourceType.FACULTY
        assert citations[3].source_type == SourceType.ANNOUNCEMENT
    
    def test_citation_clustering(self, sample_documents, sample_metadatas, sample_scores):
        """Test citation clustering"""
        tracker = CitationTracker()
        
        citations = tracker.create_citations_from_retrieval(
            sample_documents, sample_metadatas, sample_scores
        )
        
        clusters = tracker.cluster_citations(citations, similarity_threshold=0.3)
        
        assert len(clusters) > 0
        assert all(isinstance(c, CitationCluster) for c in clusters)
        # Course documents should cluster together
        assert any(len(c.citations) > 1 for c in clusters)
    
    def test_provenance_trace_creation(self, sample_documents, sample_metadatas, sample_scores):
        """Test creating provenance trace"""
        tracker = CitationTracker()
        
        citations = tracker.create_citations_from_retrieval(
            sample_documents, sample_metadatas, sample_scores
        )
        
        trace = tracker.create_provenance_trace(
            query="What are the prerequisites for EE720?",
            answer="The prerequisites for EE720 are EE301 and EE302.",
            citations=citations
        )
        
        assert isinstance(trace, ProvenanceTrace)
        assert trace.trace_id is not None
        assert len(trace.citations) == 4
        assert trace.overall_confidence > 0
        assert 0 <= trace.source_coverage <= 1
    
    def test_citation_formatting(self, sample_documents, sample_metadatas, sample_scores):
        """Test citation formatting"""
        tracker = CitationTracker()
        
        citations = tracker.create_citations_from_retrieval(
            sample_documents[:2], sample_metadatas[:2], sample_scores[:2]
        )
        
        detailed = tracker.format_citations_for_display(citations, "detailed")
        compact = tracker.format_citations_for_display(citations, "compact")
        inline = tracker.format_citations_for_display(citations, "inline")
        
        assert "Sources and Citations" in detailed
        assert "courses.json" in detailed
        
        assert "Sources:" in compact
        assert len(compact) < len(detailed)
        
        assert "Sources:" in inline
        assert len(inline) < len(compact)
    
    def test_citation_verification(self, sample_documents, sample_metadatas, sample_scores):
        """Test citation verification"""
        tracker = CitationTracker()
        
        citation = tracker.create_citation(
            content=sample_documents[0],
            metadata=sample_metadatas[0],
            relevance_score=sample_scores[0]
        )
        
        verification = tracker.verify_citation(citation)
        
        assert verification["is_valid"] is True
        assert verification["original_hash"] == verification["current_hash"]


# ============================================================
# Conflict Detector Tests
# ============================================================

class TestConflictDetector:
    """Tests for ConflictDetector"""
    
    def test_numerical_extraction(self):
        """Test numerical value extraction"""
        text = "The course has 6 credits and requires 75% attendance."
        
        extracted = NumericalExtractor.extract_numbers(text)
        
        assert 'credits' in extracted
        assert 'percentage' in extracted
        assert '6' in extracted['credits'][0]
        assert '75' in extracted['percentage'][0]
    
    def test_temporal_extraction(self):
        """Test temporal information extraction"""
        text = "The deadline is January 15, 2024. Submit before next semester."
        
        dates = TemporalExtractor.extract_dates(text)
        refs = TemporalExtractor.extract_temporal_refs(text)
        
        assert len(dates) > 0
        assert any("January" in d or "15" in d for d in dates)
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        text = "Contact Dr. Kumar at kumar@iitb.ac.in for EE720 queries. Office: Room 301"
        
        entities = EntityExtractor.extract_entities(text)
        
        assert 'email' in entities
        assert 'course_code' in entities
        assert 'room' in entities
        assert 'kumar@iitb.ac.in' in entities['email']
    
    def test_conflict_detection_no_conflicts(self, sample_documents, sample_metadatas):
        """Test conflict detection with consistent documents"""
        detector = ConflictDetector()
        
        report = detector.detect_conflicts(
            sample_documents, sample_metadatas, "Test query"
        )
        
        assert isinstance(report, ConflictReport)
        assert report.report_id is not None
        assert report.total_sources_analyzed == 4
    
    def test_conflict_detection_with_conflicts(self, conflicting_documents, conflicting_metadatas):
        """Test conflict detection with conflicting documents"""
        detector = ConflictDetector()
        
        report = detector.detect_conflicts(
            conflicting_documents, conflicting_metadatas, "What are the credits for EE720?"
        )
        
        assert not report.conflict_free
        assert len(report.conflicts) > 0
        
        # Should detect numerical conflict (6 vs 8 credits)
        num_conflicts = [c for c in report.conflicts if c.conflict_type == ConflictType.NUMERICAL]
        # Should detect attribution conflict (different instructors)
        attr_conflicts = [c for c in report.conflicts if c.conflict_type == ConflictType.ATTRIBUTION]
        
        # At least some conflicts should be detected
        assert len(report.conflicts) > 0
    
    def test_conflict_severity_assessment(self, conflicting_documents, conflicting_metadatas):
        """Test conflict severity assessment"""
        detector = ConflictDetector()
        
        report = detector.detect_conflicts(
            conflicting_documents, conflicting_metadatas
        )
        
        if report.conflicts:
            # Check severity levels are properly assigned
            for conflict in report.conflicts:
                assert conflict.severity in ConflictSeverity
                assert 0 <= conflict.detection_confidence <= 1
    
    def test_conflict_recommendations(self, conflicting_documents, conflicting_metadatas):
        """Test conflict recommendations generation"""
        detector = ConflictDetector()
        
        report = detector.detect_conflicts(
            conflicting_documents, conflicting_metadatas
        )
        
        assert len(report.recommendations) > 0
    
    def test_conflict_report_formatting(self, conflicting_documents, conflicting_metadatas):
        """Test conflict report formatting"""
        detector = ConflictDetector()
        
        report = detector.detect_conflicts(
            conflicting_documents, conflicting_metadatas
        )
        
        detailed = detector.format_conflict_report(report, "detailed")
        summary = detector.format_conflict_report(report, "summary")
        compact = detector.format_conflict_report(report, "compact")
        
        assert "CONFLICT" in detailed.upper() or "No conflicts" in detailed
        assert "Report" in summary or "conflict" in summary.lower()


# ============================================================
# Confidence Scorer Tests
# ============================================================

class TestConfidenceScorer:
    """Tests for ConfidenceScorer"""
    
    def test_calculate_confidence(self, sample_documents, sample_metadatas, sample_scores):
        """Test confidence calculation"""
        scorer = ConfidenceScorer()
        
        report = scorer.calculate_confidence(
            query="What is the prerequisite for EE720?",
            answer="The prerequisite for EE720 is EE301.",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        assert isinstance(report, ConfidenceReport)
        assert 0 <= report.breakdown.overall_score <= 1
        assert report.breakdown.confidence_level in ConfidenceLevel
        assert len(report.breakdown.component_scores) > 0
    
    def test_confidence_components(self, sample_documents, sample_metadatas, sample_scores):
        """Test individual confidence components"""
        scorer = ConfidenceScorer()
        
        report = scorer.calculate_confidence(
            query="What courses are taught by Dr. Sharma?",
            answer="Dr. Sharma teaches EE720 Advanced Power Electronics.",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        components = {cs.component: cs for cs in report.breakdown.component_scores}
        
        # Check all components are present
        assert ConfidenceComponent.RETRIEVAL_QUALITY in components
        assert ConfidenceComponent.SOURCE_AGREEMENT in components
        assert ConfidenceComponent.SOURCE_RELIABILITY in components
        assert ConfidenceComponent.ANSWER_COMPLETENESS in components
        
        # Check scores are valid
        for cs in report.breakdown.component_scores:
            assert 0 <= cs.score <= 1
            assert cs.weight > 0
    
    def test_confidence_levels(self):
        """Test confidence level determination"""
        scorer = ConfidenceScorer()
        
        # Test different score ranges
        assert scorer._determine_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert scorer._determine_level(0.80) == ConfidenceLevel.HIGH
        assert scorer._determine_level(0.60) == ConfidenceLevel.MEDIUM
        assert scorer._determine_level(0.35) == ConfidenceLevel.LOW
        assert scorer._determine_level(0.10) == ConfidenceLevel.VERY_LOW
    
    def test_quick_confidence(self):
        """Test quick confidence estimation"""
        scorer = ConfidenceScorer()
        
        # High scores, multiple sources
        high_confidence = scorer.get_quick_confidence([0.9, 0.85, 0.8], num_sources=3)
        assert high_confidence > 0.7
        
        # Low scores, single source
        low_confidence = scorer.get_quick_confidence([0.3, 0.25], num_sources=1)
        assert low_confidence < 0.5
    
    def test_confidence_with_conflicts(self, sample_documents, sample_metadatas, sample_scores):
        """Test confidence calculation with conflict report"""
        scorer = ConfidenceScorer()
        
        # Create mock conflict report
        conflict_report = {
            "conflicts": [
                {"conflict_type": "numerical", "description": "Test conflict"}
            ],
            "overall_reliability_impact": 0.2
        }
        
        report = scorer.calculate_confidence(
            query="Test query",
            answer="Test answer",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores,
            conflict_report=conflict_report
        )
        
        # Conflict should affect confidence
        assert report.breakdown.overall_score < 1.0
    
    def test_confidence_report_formatting(self, sample_documents, sample_metadatas, sample_scores):
        """Test confidence report formatting"""
        scorer = ConfidenceScorer()
        
        report = scorer.calculate_confidence(
            query="Test",
            answer="Test answer",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        detailed = scorer.format_confidence_report(report, "detailed")
        summary = scorer.format_confidence_report(report, "summary")
        compact = scorer.format_confidence_report(report, "compact")
        
        assert "Confidence" in detailed
        assert len(summary) < len(detailed)
        assert len(compact) < len(summary)


# ============================================================
# Verification Pipeline Tests
# ============================================================

class TestVerificationPipeline:
    """Tests for VerificationPipeline"""
    
    def test_full_verification(self, sample_documents, sample_metadatas, sample_scores):
        """Test full verification pipeline"""
        pipeline = VerificationPipeline()
        
        result = pipeline.verify(
            query="What are the prerequisites for EE720?",
            answer="The prerequisites for EE720 are EE301 and EE302.",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        assert isinstance(result, VerificationResult)
        assert result.result_id is not None
        assert result.summary is not None
        assert result.summary.status in VerificationStatus
        assert result.summary.risk_level in RiskLevel
        assert len(result.citations) == len(sample_documents)
    
    def test_verification_status_determination(self, sample_documents, sample_metadatas, sample_scores):
        """Test verification status determination"""
        pipeline = VerificationPipeline()
        
        result = pipeline.verify(
            query="Test query",
            answer="This is a test answer based on the provided context.",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        # With good scores and no major conflicts
        assert result.summary.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED,
            VerificationStatus.NEEDS_REVIEW
        ]
    
    def test_quick_verification(self, sample_documents, sample_metadatas, sample_scores):
        """Test quick verification"""
        pipeline = VerificationPipeline()
        
        confidence, conflict_free, citation_count = pipeline.quick_verify(
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(conflict_free, bool)
        assert citation_count == len(sample_documents)
    
    def test_verification_with_conflicts(self, conflicting_documents, conflicting_metadatas):
        """Test verification with conflicting documents"""
        pipeline = VerificationPipeline()
        
        result = pipeline.verify(
            query="What are the credits for EE720?",
            answer="EE720 has 6 credits.",
            documents=conflicting_documents,
            metadatas=conflicting_metadatas,
            retrieval_scores=[0.8, 0.75, 0.7]
        )
        
        # Should detect conflicts and potentially lower status
        if not result.conflict_report.conflict_free:
            assert result.summary.status != VerificationStatus.VERIFIED or \
                   result.summary.risk_level != RiskLevel.LOW
    
    def test_verification_result_formatting(self, sample_documents, sample_metadatas, sample_scores):
        """Test verification result formatting"""
        pipeline = VerificationPipeline()
        
        result = pipeline.verify(
            query="Test",
            answer="Test answer",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        full = pipeline.format_verification_result(result, "full")
        summary = pipeline.format_verification_result(result, "summary")
        compact = pipeline.format_verification_result(result, "compact")
        
        assert "VERIFICATION" in full.upper()
        assert "Status" in summary
        assert len(compact) < len(summary)
    
    def test_verification_stats(self, sample_documents, sample_metadatas, sample_scores):
        """Test verification statistics"""
        pipeline = VerificationPipeline()
        
        # Run a few verifications
        for _ in range(3):
            pipeline.verify(
                query="Test",
                answer="Test answer",
                documents=sample_documents,
                metadatas=sample_metadatas,
                retrieval_scores=sample_scores
            )
        
        stats = pipeline.get_verification_stats()
        
        assert stats["total_verifications"] == 3
        assert "status_distribution" in stats
        assert "average_confidence" in stats
    
    def test_verification_recommendations(self, sample_documents, sample_metadatas, sample_scores):
        """Test verification recommendations generation"""
        pipeline = VerificationPipeline()
        
        result = pipeline.verify(
            query="Test query",
            answer="Test answer",
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        assert len(result.summary.recommendations) > 0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for the complete verification system"""
    
    def test_global_instances(self):
        """Test global instance management"""
        tracker = get_citation_tracker()
        detector = get_conflict_detector()
        scorer = get_confidence_scorer()
        pipeline = get_verification_pipeline()
        
        assert tracker is not None
        assert detector is not None
        assert scorer is not None
        assert pipeline is not None
        
        # Should return same instances
        assert get_citation_tracker() is tracker
        assert get_conflict_detector() is detector
        assert get_confidence_scorer() is scorer
        assert get_verification_pipeline() is pipeline
    
    def test_convenience_functions(self, sample_documents, sample_metadatas, sample_scores):
        """Test convenience functions"""
        # Citation tracking
        citations = create_citations_from_retrieval(
            sample_documents, sample_metadatas, sample_scores
        )
        assert len(citations) == 4
        
        # Conflict detection
        conflict_report = detect_conflicts(
            sample_documents, sample_metadatas, "Test query"
        )
        assert conflict_report is not None
        
        # Confidence scoring
        confidence_report = calculate_confidence(
            "Test query", "Test answer",
            sample_documents, sample_metadatas, sample_scores
        )
        assert confidence_report is not None
        
        # Full verification
        result = verify_response(
            "Test query", "Test answer",
            sample_documents, sample_metadatas, sample_scores
        )
        assert result is not None
        
        # Quick verification
        conf, no_conflicts, count = quick_verify(
            sample_documents, sample_metadatas, sample_scores
        )
        assert 0 <= conf <= 1
    
    def test_end_to_end_pipeline(self, sample_documents, sample_metadatas, sample_scores):
        """Test complete end-to-end verification"""
        # Simulate a RAG query flow
        query = "What are the prerequisites for EE720 and who teaches it?"
        answer = "EE720 (Advanced Power Electronics) has prerequisites EE301 and EE302. It is taught by Dr. R.K. Sharma."
        
        # Run verification
        result = verify_response(
            query=query,
            answer=answer,
            documents=sample_documents,
            metadatas=sample_metadatas,
            retrieval_scores=sample_scores
        )
        
        # Check all components are populated
        assert result.citations is not None
        assert result.provenance_trace is not None
        assert result.conflict_report is not None
        assert result.confidence_report is not None
        
        # Check summary
        assert result.summary.confidence_score > 0
        assert result.summary.citation_count == 4
        assert result.summary.source_count >= 1
        
        # Verify formatting works
        formatted = get_verification_pipeline().format_verification_result(result, "full")
        assert len(formatted) > 0
    
    def test_serialization(self, sample_documents, sample_metadatas, sample_scores):
        """Test result serialization"""
        result = verify_response(
            "Test", "Test answer",
            sample_documents, sample_metadatas, sample_scores
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        assert "result_id" in result_dict
        assert "summary" in result_dict
        
        # Export to JSON
        pipeline = get_verification_pipeline()
        json_str = pipeline.export_result(result)
        assert len(json_str) > 0
        
        # Verify it's valid JSON
        import json
        parsed = json.loads(json_str)
        assert parsed["result_id"] == result.result_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
