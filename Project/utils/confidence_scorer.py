"""
Confidence Scoring Module
This module provides comprehensive confidence scoring for RAG outputs based on
retrieval quality, source agreement, citation reliability, and multiple other factors.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict
import re
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Categorical confidence levels"""
    VERY_HIGH = "very_high"   # > 90%
    HIGH = "high"             # 75-90%
    MEDIUM = "medium"         # 50-75%
    LOW = "low"               # 25-50%
    VERY_LOW = "very_low"     # < 25%
    UNCERTAIN = "uncertain"   # Cannot determine


class ConfidenceComponent(Enum):
    """Components that contribute to confidence"""
    RETRIEVAL_QUALITY = "retrieval_quality"
    SOURCE_AGREEMENT = "source_agreement"
    CITATION_COVERAGE = "citation_coverage"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SOURCE_RELIABILITY = "source_reliability"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    QUERY_SPECIFICITY = "query_specificity"
    ANSWER_COMPLETENESS = "answer_completeness"
    CONFLICT_PENALTY = "conflict_penalty"


@dataclass
class ComponentScore:
    """Score for a single confidence component"""
    component: ConfidenceComponent
    score: float  # 0 to 1
    weight: float  # Weight in final calculation
    explanation: str
    contributing_factors: List[str] = field(default_factory=list)
    
    def weighted_score(self) -> float:
        """Get weighted score"""
        return self.score * self.weight


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scoring"""
    component_scores: List[ComponentScore]
    overall_score: float
    confidence_level: ConfidenceLevel
    explanation: str
    warnings: List[str]
    strengths: List[str]
    weaknesses: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "confidence_level": self.confidence_level.value,
            "explanation": self.explanation,
            "component_scores": [
                {
                    "component": cs.component.value,
                    "score": cs.score,
                    "weight": cs.weight,
                    "weighted_score": cs.weighted_score(),
                    "explanation": cs.explanation
                }
                for cs in self.component_scores
            ],
            "warnings": self.warnings,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses
        }


@dataclass
class ConfidenceReport:
    """Complete confidence report for a RAG response"""
    report_id: str
    query: str
    answer: str
    breakdown: ConfidenceBreakdown
    metadata: Dict[str, Any]
    generated_at: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": self.report_id,
            "query": self.query,
            "answer_preview": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "breakdown": self.breakdown.to_dict(),
            "metadata": self.metadata,
            "generated_at": self.generated_at,
            "recommendations": self.recommendations
        }


class RetrievalQualityScorer:
    """Scores the quality of retrieval results"""
    
    def __init__(self):
        self.score_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "fair": 0.50,
            "poor": 0.30
        }
    
    def score(
        self,
        retrieval_scores: List[float],
        query: str,
        documents: List[str]
    ) -> ComponentScore:
        """
        Score retrieval quality.
        
        Args:
            retrieval_scores: Relevance scores from retrieval
            query: Original query
            documents: Retrieved documents
            
        Returns:
            ComponentScore for retrieval quality
        """
        if not retrieval_scores:
            return ComponentScore(
                component=ConfidenceComponent.RETRIEVAL_QUALITY,
                score=0.0,
                weight=0.25,
                explanation="No retrieval results available",
                contributing_factors=["No documents retrieved"]
            )
        
        factors = []
        
        # 1. Average retrieval score
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        factors.append(f"Average relevance: {avg_score:.2f}")
        
        # 2. Top-k quality (top results should be high quality)
        top_k = min(3, len(retrieval_scores))
        top_avg = sum(retrieval_scores[:top_k]) / top_k
        factors.append(f"Top-{top_k} average: {top_avg:.2f}")
        
        # 3. Score distribution (prefer consistent high scores)
        if len(retrieval_scores) > 1:
            score_variance = sum(
                (s - avg_score) ** 2 for s in retrieval_scores
            ) / len(retrieval_scores)
            variance_penalty = min(score_variance * 0.5, 0.2)
            factors.append(f"Score variance: {score_variance:.3f}")
        else:
            variance_penalty = 0
        
        # 4. Query coverage (how well documents cover query terms)
        query_terms = set(query.lower().split())
        coverage_scores = []
        for doc in documents:
            doc_terms = set(doc.lower().split())
            if query_terms:
                coverage = len(query_terms & doc_terms) / len(query_terms)
                coverage_scores.append(coverage)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        factors.append(f"Query term coverage: {avg_coverage:.2%}")
        
        # 5. Document diversity
        unique_content_ratio = len(set(d[:100] for d in documents)) / len(documents)
        factors.append(f"Content diversity: {unique_content_ratio:.2%}")
        
        # Calculate final score
        base_score = (top_avg * 0.5 + avg_score * 0.3 + avg_coverage * 0.2)
        final_score = max(0, min(1, base_score - variance_penalty))
        
        # Generate explanation
        if final_score >= self.score_thresholds["excellent"]:
            explanation = "Excellent retrieval quality with highly relevant results"
        elif final_score >= self.score_thresholds["good"]:
            explanation = "Good retrieval quality with relevant results"
        elif final_score >= self.score_thresholds["fair"]:
            explanation = "Fair retrieval quality, some results may be less relevant"
        else:
            explanation = "Low retrieval quality, results may not fully address the query"
        
        return ComponentScore(
            component=ConfidenceComponent.RETRIEVAL_QUALITY,
            score=final_score,
            weight=0.25,
            explanation=explanation,
            contributing_factors=factors
        )


class SourceAgreementScorer:
    """Scores agreement between multiple sources"""
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
    
    def score(
        self,
        documents: List[str],
        metadatas: List[Dict]
    ) -> ComponentScore:
        """
        Score source agreement.
        
        Args:
            documents: Retrieved documents
            metadatas: Document metadata
            
        Returns:
            ComponentScore for source agreement
        """
        if len(documents) <= 1:
            return ComponentScore(
                component=ConfidenceComponent.SOURCE_AGREEMENT,
                score=0.7,  # Single source, moderate confidence
                weight=0.20,
                explanation="Single source - cannot assess agreement",
                contributing_factors=["Only one source available"]
            )
        
        factors = []
        
        # 1. Pairwise content agreement
        agreement_scores = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                sim = self._compute_similarity(documents[i], documents[j])
                agreement_scores.append(sim)
        
        avg_agreement = sum(agreement_scores) / len(agreement_scores)
        factors.append(f"Average content agreement: {avg_agreement:.2%}")
        
        # 2. Source diversity (different sources agreeing is stronger)
        unique_sources = len(set(m.get('source', f'src_{i}') for i, m in enumerate(metadatas)))
        diversity_ratio = unique_sources / len(documents)
        factors.append(f"Source diversity: {unique_sources} unique sources")
        
        # 3. Key fact consistency
        key_facts_consistency = self._check_key_facts_consistency(documents)
        factors.append(f"Key fact consistency: {key_facts_consistency:.2%}")
        
        # 4. High agreement from diverse sources is best
        if diversity_ratio > 0.7 and avg_agreement > 0.6:
            diversity_bonus = 0.15
            factors.append("Bonus: Diverse sources showing agreement")
        else:
            diversity_bonus = 0
        
        # Calculate final score
        final_score = min(1.0, (
            avg_agreement * 0.4 +
            key_facts_consistency * 0.4 +
            diversity_ratio * 0.2 +
            diversity_bonus
        ))
        
        # Generate explanation
        if final_score >= 0.8:
            explanation = "Strong agreement across multiple sources"
        elif final_score >= 0.6:
            explanation = "Moderate agreement between sources"
        elif final_score >= 0.4:
            explanation = "Some disagreement between sources"
        else:
            explanation = "Significant disagreement or inconsistency between sources"
        
        return ComponentScore(
            component=ConfidenceComponent.SOURCE_AGREEMENT,
            score=final_score,
            weight=0.20,
            explanation=explanation,
            contributing_factors=factors
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_key_facts_consistency(self, documents: List[str]) -> float:
        """Check consistency of key facts across documents"""
        # Extract numerical values and named entities
        all_numbers = []
        all_codes = []
        
        for doc in documents:
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', doc)
            codes = re.findall(r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?', doc)
            all_numbers.extend(numbers)
            all_codes.extend(codes)
        
        if not all_numbers and not all_codes:
            return 0.7  # No key facts to compare
        
        # Check for consistency (same values appearing multiple times)
        number_consistency = 0.0
        if all_numbers:
            from collections import Counter
            number_counts = Counter(all_numbers)
            repeated = sum(1 for c in number_counts.values() if c > 1)
            number_consistency = repeated / len(number_counts) if number_counts else 0
        
        code_consistency = 0.0
        if all_codes:
            from collections import Counter
            code_counts = Counter(all_codes)
            repeated = sum(1 for c in code_counts.values() if c > 1)
            code_consistency = repeated / len(code_counts) if code_counts else 0
        
        return (number_consistency + code_consistency) / 2 if (all_numbers or all_codes) else 0.7


class SourceReliabilityScorer:
    """Scores the reliability of sources"""
    
    def __init__(self):
        # Source type reliability ratings
        self.type_reliability = {
            'course': 0.95,
            'faculty': 0.90,
            'announcement': 0.85,
            'research': 0.80,
            'general': 0.70,
            'unknown': 0.50
        }
    
    def score(self, metadatas: List[Dict]) -> ComponentScore:
        """
        Score source reliability.
        
        Args:
            metadatas: Document metadata
            
        Returns:
            ComponentScore for source reliability
        """
        if not metadatas:
            return ComponentScore(
                component=ConfidenceComponent.SOURCE_RELIABILITY,
                score=0.5,
                weight=0.15,
                explanation="No source metadata available",
                contributing_factors=["Missing metadata"]
            )
        
        factors = []
        
        # 1. Source type reliability
        type_scores = []
        for meta in metadatas:
            source_type = meta.get('source_type', 'unknown')
            reliability = self.type_reliability.get(source_type, 0.5)
            type_scores.append(reliability)
        
        avg_type_score = sum(type_scores) / len(type_scores)
        factors.append(f"Average source type reliability: {avg_type_score:.2%}")
        
        # 2. Temporal freshness
        freshness_scores = []
        for meta in metadatas:
            timestamp = meta.get('timestamp')
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp)
                    age_days = (datetime.now() - ts).days
                    freshness = max(0, 1 - (age_days / 365))  # Decay over a year
                    freshness_scores.append(freshness)
                except (ValueError, TypeError):
                    freshness_scores.append(0.5)
            else:
                freshness_scores.append(0.5)
        
        avg_freshness = sum(freshness_scores) / len(freshness_scores)
        factors.append(f"Average freshness: {avg_freshness:.2%}")
        
        # 3. Source completeness (has expected metadata)
        completeness_scores = []
        expected_fields = ['source', 'source_type', 'doc_id', 'timestamp']
        for meta in metadatas:
            present = sum(1 for f in expected_fields if f in meta and meta[f])
            completeness_scores.append(present / len(expected_fields))
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        factors.append(f"Metadata completeness: {avg_completeness:.2%}")
        
        # Calculate final score
        final_score = (
            avg_type_score * 0.5 +
            avg_freshness * 0.3 +
            avg_completeness * 0.2
        )
        
        # Generate explanation
        source_types = set(m.get('source_type', 'unknown') for m in metadatas)
        if final_score >= 0.85:
            explanation = f"Highly reliable sources: {', '.join(source_types)}"
        elif final_score >= 0.70:
            explanation = f"Reliable sources from: {', '.join(source_types)}"
        elif final_score >= 0.50:
            explanation = "Moderately reliable sources"
        else:
            explanation = "Source reliability is uncertain"
        
        return ComponentScore(
            component=ConfidenceComponent.SOURCE_RELIABILITY,
            score=final_score,
            weight=0.15,
            explanation=explanation,
            contributing_factors=factors
        )


class AnswerCompletenessScorer:
    """Scores how completely the answer addresses the query"""
    
    def score(
        self,
        query: str,
        answer: str,
        documents: List[str]
    ) -> ComponentScore:
        """
        Score answer completeness.
        
        Args:
            query: Original query
            answer: Generated answer
            documents: Source documents
            
        Returns:
            ComponentScore for answer completeness
        """
        factors = []
        
        # 1. Query term coverage in answer
        query_terms = self._extract_significant_terms(query)
        answer_terms = set(answer.lower().split())
        
        if query_terms:
            query_coverage = len(query_terms & answer_terms) / len(query_terms)
            factors.append(f"Query term coverage: {query_coverage:.2%}")
        else:
            query_coverage = 0.5
        
        # 2. Answer length appropriateness
        answer_words = len(answer.split())
        query_complexity = self._estimate_query_complexity(query)
        
        # Expected length based on complexity
        min_length = query_complexity * 20
        max_length = query_complexity * 200
        
        if min_length <= answer_words <= max_length:
            length_score = 1.0
            factors.append(f"Appropriate answer length: {answer_words} words")
        elif answer_words < min_length:
            length_score = answer_words / min_length
            factors.append(f"Answer may be too brief: {answer_words} words")
        else:
            length_score = max(0.5, 1 - (answer_words - max_length) / max_length)
            factors.append(f"Answer may be too verbose: {answer_words} words")
        
        # 3. Source utilization
        source_terms = set()
        for doc in documents:
            source_terms.update(doc.lower().split())
        
        answer_from_sources = len(answer_terms & source_terms) / len(answer_terms) if answer_terms else 0
        factors.append(f"Answer grounded in sources: {answer_from_sources:.2%}")
        
        # 4. Check for uncertainty indicators
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", "cannot determine",
            "no information", "not found", "not available"
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        if has_uncertainty:
            uncertainty_penalty = 0.2
            factors.append("Contains uncertainty indicators")
        else:
            uncertainty_penalty = 0
        
        # 5. Check for specific answer components
        has_specifics = bool(re.search(r'\d+|[A-Z]{2,4}\d{3,4}', answer))
        specificity_bonus = 0.1 if has_specifics else 0
        factors.append(f"Contains specific details: {has_specifics}")
        
        # Calculate final score
        final_score = max(0, min(1, (
            query_coverage * 0.3 +
            length_score * 0.2 +
            answer_from_sources * 0.3 +
            0.2 +  # Base score
            specificity_bonus -
            uncertainty_penalty
        )))
        
        # Generate explanation
        if final_score >= 0.8:
            explanation = "Answer comprehensively addresses the query"
        elif final_score >= 0.6:
            explanation = "Answer adequately addresses main aspects of the query"
        elif final_score >= 0.4:
            explanation = "Answer partially addresses the query"
        else:
            explanation = "Answer may not fully address the query"
        
        return ComponentScore(
            component=ConfidenceComponent.ANSWER_COMPLETENESS,
            score=final_score,
            weight=0.15,
            explanation=explanation,
            contributing_factors=factors
        )
    
    def _extract_significant_terms(self, text: str) -> Set[str]:
        """Extract significant terms from text"""
        words = set(text.lower().split())
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'what', 'which', 'who',
            'how', 'when', 'where', 'why', 'and', 'or', 'but', 'if', 'then',
            'else', 'me', 'my', 'your', 'it', 'this', 'that', 'these', 'those'
        }
        return words - stopwords
    
    def _estimate_query_complexity(self, query: str) -> int:
        """Estimate query complexity (1-5)"""
        words = len(query.split())
        
        # Check for complex question patterns
        complex_patterns = [
            r'\band\b.*\band\b',  # Multiple parts
            r'compare|difference|similarity',
            r'why|how|explain',
            r'all|every|each',
        ]
        
        complexity = 1
        
        if words > 15:
            complexity += 1
        if words > 30:
            complexity += 1
        
        for pattern in complex_patterns:
            if re.search(pattern, query.lower()):
                complexity += 1
        
        return min(complexity, 5)


class SemanticSimilarityScorer:
    """Scores semantic similarity between query, answer, and sources"""
    
    def score(
        self,
        query: str,
        answer: str,
        documents: List[str]
    ) -> ComponentScore:
        """
        Score semantic similarity.
        
        Args:
            query: Original query
            answer: Generated answer
            documents: Source documents
            
        Returns:
            ComponentScore for semantic similarity
        """
        factors = []
        
        # 1. Query-Answer similarity
        qa_similarity = self._compute_similarity(query, answer)
        factors.append(f"Query-Answer similarity: {qa_similarity:.2%}")
        
        # 2. Answer-Source similarity
        as_similarities = [
            self._compute_similarity(answer, doc)
            for doc in documents
        ]
        avg_as_similarity = sum(as_similarities) / len(as_similarities) if as_similarities else 0
        factors.append(f"Answer-Source similarity: {avg_as_similarity:.2%}")
        
        # 3. Semantic coherence (answer should relate more to query than random)
        coherence_score = min(1.0, qa_similarity * 1.5)
        factors.append(f"Semantic coherence: {coherence_score:.2%}")
        
        # Calculate final score
        final_score = (
            qa_similarity * 0.3 +
            avg_as_similarity * 0.5 +
            coherence_score * 0.2
        )
        
        # Generate explanation
        if final_score >= 0.7:
            explanation = "Strong semantic alignment between query, answer, and sources"
        elif final_score >= 0.5:
            explanation = "Moderate semantic alignment"
        else:
            explanation = "Weak semantic alignment - answer may be tangential"
        
        return ComponentScore(
            component=ConfidenceComponent.SEMANTIC_SIMILARITY,
            score=final_score,
            weight=0.10,
            explanation=explanation,
            contributing_factors=factors
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using term overlap with IDF-like weighting"""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        if not words1 or not words2:
            return 0.0
        
        # Use TF-IDF-like approach
        all_words = set(words1) | set(words2)
        
        # Simple IDF: rarer words get higher weight
        word_freq = defaultdict(int)
        for w in words1 + words2:
            word_freq[w] += 1
        
        max_freq = max(word_freq.values())
        idf_weights = {w: 1 - (f / max_freq) + 0.5 for w, f in word_freq.items()}
        
        # Weighted intersection
        set1 = set(words1)
        set2 = set(words2)
        
        intersection = set1 & set2
        weighted_intersection = sum(idf_weights.get(w, 1) for w in intersection)
        weighted_union = sum(idf_weights.get(w, 1) for w in (set1 | set2))
        
        return weighted_intersection / weighted_union if weighted_union > 0 else 0.0


class ConflictPenaltyScorer:
    """Applies confidence penalty based on detected conflicts"""
    
    def score(
        self,
        conflict_report: Optional[Dict] = None,
        num_conflicts: int = 0,
        reliability_impact: float = 0.0
    ) -> ComponentScore:
        """
        Score conflict penalty.
        
        Args:
            conflict_report: Optional conflict detection report
            num_conflicts: Number of detected conflicts
            reliability_impact: Overall reliability impact from conflicts
            
        Returns:
            ComponentScore for conflict penalty (inverse - higher means less conflicts)
        """
        factors = []
        
        if conflict_report:
            num_conflicts = conflict_report.get('conflicts', [])
            num_conflicts = len(num_conflicts) if isinstance(num_conflicts, list) else num_conflicts
            reliability_impact = conflict_report.get('overall_reliability_impact', 0)
        
        factors.append(f"Number of conflicts: {num_conflicts}")
        factors.append(f"Reliability impact: {reliability_impact:.2%}")
        
        # Calculate penalty
        if num_conflicts == 0:
            penalty = 0.0
            explanation = "No conflicts detected"
        elif num_conflicts <= 2:
            penalty = 0.1 + reliability_impact * 0.3
            explanation = f"Minor conflicts detected ({num_conflicts})"
        elif num_conflicts <= 5:
            penalty = 0.25 + reliability_impact * 0.4
            explanation = f"Moderate conflicts detected ({num_conflicts})"
        else:
            penalty = 0.4 + reliability_impact * 0.5
            explanation = f"Significant conflicts detected ({num_conflicts})"
        
        # Score is inverse of penalty (1 - penalty = no conflicts is good)
        final_score = max(0, 1 - penalty)
        
        return ComponentScore(
            component=ConfidenceComponent.CONFLICT_PENALTY,
            score=final_score,
            weight=0.10,
            explanation=explanation,
            contributing_factors=factors
        )


class ConfidenceScorer:
    """
    Comprehensive confidence scoring system for RAG outputs.
    Combines multiple scoring components to provide an overall confidence assessment.
    """
    
    def __init__(self):
        """Initialize the confidence scorer with all component scorers"""
        self.retrieval_scorer = RetrievalQualityScorer()
        self.agreement_scorer = SourceAgreementScorer()
        self.reliability_scorer = SourceReliabilityScorer()
        self.completeness_scorer = AnswerCompletenessScorer()
        self.similarity_scorer = SemanticSimilarityScorer()
        self.conflict_scorer = ConflictPenaltyScorer()
        
        self.score_history: List[ConfidenceReport] = []
        
        # Confidence level thresholds
        self.level_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.90,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.MEDIUM: 0.50,
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.VERY_LOW: 0.0
        }
    
    def calculate_confidence(
        self,
        query: str,
        answer: str,
        documents: List[str],
        metadatas: List[Dict],
        retrieval_scores: List[float],
        conflict_report: Optional[Dict] = None
    ) -> ConfidenceReport:
        """
        Calculate comprehensive confidence score for a RAG response.
        
        Args:
            query: Original user query
            answer: Generated answer
            documents: Retrieved documents
            metadatas: Document metadata
            retrieval_scores: Relevance scores from retrieval
            conflict_report: Optional conflict detection report
            
        Returns:
            Complete ConfidenceReport with detailed breakdown
        """
        component_scores = []
        
        # 1. Retrieval Quality
        retrieval_score = self.retrieval_scorer.score(
            retrieval_scores, query, documents
        )
        component_scores.append(retrieval_score)
        
        # 2. Source Agreement
        agreement_score = self.agreement_scorer.score(documents, metadatas)
        component_scores.append(agreement_score)
        
        # 3. Source Reliability
        reliability_score = self.reliability_scorer.score(metadatas)
        component_scores.append(reliability_score)
        
        # 4. Answer Completeness
        completeness_score = self.completeness_scorer.score(query, answer, documents)
        component_scores.append(completeness_score)
        
        # 5. Semantic Similarity
        similarity_score = self.similarity_scorer.score(query, answer, documents)
        component_scores.append(similarity_score)
        
        # 6. Conflict Penalty
        conflict_score = self.conflict_scorer.score(conflict_report=conflict_report)
        component_scores.append(conflict_score)
        
        # Calculate overall score
        total_weight = sum(cs.weight for cs in component_scores)
        overall_score = sum(cs.weighted_score() for cs in component_scores) / total_weight
        
        # Determine confidence level
        confidence_level = self._determine_level(overall_score)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(component_scores)
        
        # Generate warnings
        warnings = self._generate_warnings(component_scores, overall_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_score, confidence_level, component_scores
        )
        
        # Create breakdown
        breakdown = ConfidenceBreakdown(
            component_scores=component_scores,
            overall_score=overall_score,
            confidence_level=confidence_level,
            explanation=explanation,
            warnings=warnings,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(breakdown)
        
        # Create report
        report = ConfidenceReport(
            report_id=hashlib.md5(
                f"{query}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            query=query,
            answer=answer,
            breakdown=breakdown,
            metadata={
                "num_documents": len(documents),
                "num_sources": len(set(m.get('source', f's{i}') for i, m in enumerate(metadatas))),
                "answer_length": len(answer.split()),
                "query_length": len(query.split())
            },
            generated_at=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
        self.score_history.append(report)
        
        logger.info(
            f"Confidence calculated: {overall_score:.2%} ({confidence_level.value})"
        )
        
        return report
    
    def _determine_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score"""
        for level, threshold in self.level_thresholds.items():
            if score >= threshold:
                return level
        return ConfidenceLevel.UNCERTAIN
    
    def _identify_strengths_weaknesses(
        self,
        component_scores: List[ComponentScore]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses from component scores"""
        strengths = []
        weaknesses = []
        
        for cs in component_scores:
            if cs.score >= 0.8:
                strengths.append(f"{cs.component.value}: {cs.explanation}")
            elif cs.score <= 0.4:
                weaknesses.append(f"{cs.component.value}: {cs.explanation}")
        
        return strengths, weaknesses
    
    def _generate_warnings(
        self,
        component_scores: List[ComponentScore],
        overall_score: float
    ) -> List[str]:
        """Generate warning messages"""
        warnings = []
        
        # Check for low individual scores
        for cs in component_scores:
            if cs.score < 0.3:
                warnings.append(
                    f"⚠️ Low {cs.component.value.replace('_', ' ')}: {cs.score:.2%}"
                )
        
        # Overall confidence warnings
        if overall_score < 0.5:
            warnings.append("⚠️ Overall confidence is below recommended threshold")
        
        # Check for high variance in scores
        scores = [cs.score for cs in component_scores]
        if scores:
            variance = sum((s - overall_score) ** 2 for s in scores) / len(scores)
            if variance > 0.1:
                warnings.append("⚠️ High variance in confidence components")
        
        return warnings
    
    def _generate_explanation(
        self,
        overall_score: float,
        confidence_level: ConfidenceLevel,
        component_scores: List[ComponentScore]
    ) -> str:
        """Generate overall explanation"""
        level_descriptions = {
            ConfidenceLevel.VERY_HIGH: "Very high confidence - the answer is well-supported by multiple reliable sources",
            ConfidenceLevel.HIGH: "High confidence - the answer is supported by good quality sources",
            ConfidenceLevel.MEDIUM: "Moderate confidence - the answer is partially supported but may need verification",
            ConfidenceLevel.LOW: "Low confidence - the answer may be incomplete or insufficiently supported",
            ConfidenceLevel.VERY_LOW: "Very low confidence - the answer should be verified from other sources",
            ConfidenceLevel.UNCERTAIN: "Uncertain - confidence could not be reliably determined"
        }
        
        base_explanation = level_descriptions.get(
            confidence_level,
            "Confidence level could not be determined"
        )
        
        # Add specific notes
        highest = max(component_scores, key=lambda x: x.score)
        lowest = min(component_scores, key=lambda x: x.score)
        
        return (
            f"{base_explanation}. "
            f"Strongest factor: {highest.component.value.replace('_', ' ')} ({highest.score:.0%}). "
            f"Weakest factor: {lowest.component.value.replace('_', ' ')} ({lowest.score:.0%})."
        )
    
    def _generate_recommendations(
        self,
        breakdown: ConfidenceBreakdown
    ) -> List[str]:
        """Generate recommendations based on confidence analysis"""
        recommendations = []
        
        # Based on confidence level
        if breakdown.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
            recommendations.append(
                "Consider verifying this information from official sources"
            )
        
        # Based on specific weaknesses
        for cs in breakdown.component_scores:
            if cs.component == ConfidenceComponent.RETRIEVAL_QUALITY and cs.score < 0.5:
                recommendations.append(
                    "Try rephrasing your query for better retrieval results"
                )
            elif cs.component == ConfidenceComponent.SOURCE_AGREEMENT and cs.score < 0.5:
                recommendations.append(
                    "Sources show disagreement - verify critical details"
                )
            elif cs.component == ConfidenceComponent.CONFLICT_PENALTY and cs.score < 0.7:
                recommendations.append(
                    "Conflicts detected - check conflict report for details"
                )
        
        if not recommendations:
            recommendations.append(
                "Confidence is adequate - answer can be trusted for general purposes"
            )
        
        return recommendations
    
    def format_confidence_report(
        self,
        report: ConfidenceReport,
        format_type: str = "detailed"
    ) -> str:
        """
        Format confidence report for display.
        
        Args:
            report: Confidence report to format
            format_type: "detailed", "summary", or "compact"
            
        Returns:
            Formatted string
        """
        if format_type == "summary":
            return self._format_summary(report)
        elif format_type == "compact":
            return self._format_compact(report)
        else:
            return self._format_detailed(report)
    
    def _format_detailed(self, report: ConfidenceReport) -> str:
        """Format detailed confidence report"""
        breakdown = report.breakdown
        
        # Emoji for confidence level
        level_emoji = {
            ConfidenceLevel.VERY_HIGH: "🟢",
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "🔴",
            ConfidenceLevel.UNCERTAIN: "⚪"
        }
        
        lines = [
            "═" * 60,
            "📊 **CONFIDENCE ANALYSIS REPORT**",
            "═" * 60,
            f"Report ID: {report.report_id}",
            f"Generated: {report.generated_at}",
            "",
            f"**Overall Confidence: {breakdown.overall_score:.1%}** "
            f"{level_emoji.get(breakdown.confidence_level, '')} "
            f"({breakdown.confidence_level.value.replace('_', ' ').title()})",
            "",
            breakdown.explanation,
            "",
            "**Component Scores:**",
            "-" * 40
        ]
        
        # Add component scores
        for cs in breakdown.component_scores:
            bar_length = int(cs.score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(
                f"  {cs.component.value.replace('_', ' ').title():25} "
                f"[{bar}] {cs.score:.0%} (w:{cs.weight:.0%})"
            )
        
        lines.append("")
        
        # Strengths
        if breakdown.strengths:
            lines.append("**💪 Strengths:**")
            for s in breakdown.strengths:
                lines.append(f"  ✓ {s}")
            lines.append("")
        
        # Weaknesses
        if breakdown.weaknesses:
            lines.append("**⚠️ Weaknesses:**")
            for w in breakdown.weaknesses:
                lines.append(f"  ✗ {w}")
            lines.append("")
        
        # Warnings
        if breakdown.warnings:
            lines.append("**⚡ Warnings:**")
            for w in breakdown.warnings:
                lines.append(f"  {w}")
            lines.append("")
        
        # Recommendations
        lines.append("**💡 Recommendations:**")
        for r in report.recommendations:
            lines.append(f"  • {r}")
        
        return "\n".join(lines)
    
    def _format_summary(self, report: ConfidenceReport) -> str:
        """Format summary confidence report"""
        breakdown = report.breakdown
        return (
            f"Confidence: {breakdown.overall_score:.1%} "
            f"({breakdown.confidence_level.value.replace('_', ' ').title()}) | "
            f"Sources: {report.metadata.get('num_sources', '?')} | "
            f"Warnings: {len(breakdown.warnings)}"
        )
    
    def _format_compact(self, report: ConfidenceReport) -> str:
        """Format compact confidence report"""
        breakdown = report.breakdown
        
        level_indicator = {
            ConfidenceLevel.VERY_HIGH: "🟢",
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "🔴",
            ConfidenceLevel.UNCERTAIN: "⚪"
        }
        
        return (
            f"{level_indicator.get(breakdown.confidence_level, '⚪')} "
            f"Confidence: {breakdown.overall_score:.0%}"
        )
    
    def get_quick_confidence(
        self,
        retrieval_scores: List[float],
        num_sources: int = 1
    ) -> float:
        """
        Get a quick confidence estimate from retrieval scores.
        
        Args:
            retrieval_scores: Relevance scores from retrieval
            num_sources: Number of unique sources
            
        Returns:
            Quick confidence estimate (0-1)
        """
        if not retrieval_scores:
            return 0.3
        
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        diversity_bonus = min(num_sources * 0.05, 0.2)
        
        return min(1.0, avg_score * 0.8 + diversity_bonus + 0.1)
    
    def get_score_history(self, limit: int = 10) -> List[ConfidenceReport]:
        """Get recent confidence reports"""
        return self.score_history[-limit:]
    
    def export_report(self, report: ConfidenceReport) -> str:
        """Export report as JSON"""
        return json.dumps(report.to_dict(), indent=2)


# Global instance
_confidence_scorer = None

def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create confidence scorer instance"""
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


def calculate_confidence(
    query: str,
    answer: str,
    documents: List[str],
    metadatas: List[Dict],
    retrieval_scores: List[float],
    conflict_report: Optional[Dict] = None
) -> ConfidenceReport:
    """
    Convenience function to calculate confidence.
    
    Args:
        query: Original query
        answer: Generated answer
        documents: Retrieved documents
        metadatas: Document metadata
        retrieval_scores: Retrieval scores
        conflict_report: Optional conflict report
        
    Returns:
        ConfidenceReport
    """
    scorer = get_confidence_scorer()
    return scorer.calculate_confidence(
        query, answer, documents, metadatas, retrieval_scores, conflict_report
    )


def get_quick_confidence(
    retrieval_scores: List[float],
    num_sources: int = 1
) -> float:
    """
    Convenience function for quick confidence estimation.
    
    Args:
        retrieval_scores: Relevance scores
        num_sources: Number of unique sources
        
    Returns:
        Quick confidence estimate
    """
    scorer = get_confidence_scorer()
    return scorer.get_quick_confidence(retrieval_scores, num_sources)
