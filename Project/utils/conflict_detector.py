"""
Conflict Detection Module
This module identifies contradictory, inconsistent, or conflicting information
across different sources in the RAG system.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can be detected"""
    FACTUAL = "factual"           # Direct contradiction of facts
    TEMPORAL = "temporal"          # Conflicting time-related info
    NUMERICAL = "numerical"        # Conflicting numbers/statistics
    ATTRIBUTION = "attribution"    # Different attributions for same thing
    COMPLETENESS = "completeness"  # Missing vs present information
    SEMANTIC = "semantic"          # Semantically contradictory statements
    OUTDATED = "outdated"          # Information that may be outdated


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    CRITICAL = "critical"   # Major contradiction affecting answer reliability
    HIGH = "high"           # Significant conflict needing attention
    MEDIUM = "medium"       # Notable inconsistency
    LOW = "low"             # Minor discrepancy
    INFO = "info"           # Informational, not a real conflict


@dataclass
class ConflictEvidence:
    """Evidence supporting a detected conflict"""
    source_file: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None
    relevance_to_conflict: float = 1.0


@dataclass
class DetectedConflict:
    """Represents a detected conflict between sources"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    evidence_a: ConflictEvidence
    evidence_b: ConflictEvidence
    conflicting_claims: Tuple[str, str]
    detection_confidence: float
    resolution_suggestion: Optional[str] = None
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence_a": {
                "source_file": self.evidence_a.source_file,
                "document_id": self.evidence_a.document_id,
                "content": self.evidence_a.content[:200],
                "timestamp": self.evidence_a.timestamp
            },
            "evidence_b": {
                "source_file": self.evidence_b.source_file,
                "document_id": self.evidence_b.document_id,
                "content": self.evidence_b.content[:200],
                "timestamp": self.evidence_b.timestamp
            },
            "conflicting_claims": self.conflicting_claims,
            "detection_confidence": self.detection_confidence,
            "resolution_suggestion": self.resolution_suggestion,
            "detected_at": self.detected_at,
            "resolved": self.resolved
        }


@dataclass
class ConflictReport:
    """Complete report of conflicts for a query"""
    report_id: str
    query: str
    conflicts: List[DetectedConflict]
    total_sources_analyzed: int
    conflict_free: bool
    overall_reliability_impact: float
    generated_at: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": self.report_id,
            "query": self.query,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "total_sources_analyzed": self.total_sources_analyzed,
            "conflict_free": self.conflict_free,
            "overall_reliability_impact": self.overall_reliability_impact,
            "generated_at": self.generated_at,
            "recommendations": self.recommendations
        }


class NumericalExtractor:
    """Extracts and normalizes numerical values from text"""
    
    # Patterns for common numerical expressions
    PATTERNS = {
        'credits': r'(\d+)\s*(?:credits?|cr\.?)',
        'percentage': r'(\d+(?:\.\d+)?)\s*%',
        'years': r'(\d+)\s*(?:years?|yrs?)',
        'count': r'(\d+)\s*(?:students?|courses?|faculty|members?)',
        'time': r'(\d{1,2}):(\d{2})\s*(?:am|pm)?',
        'date': r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        'phone': r'\+?\d[\d\s-]{8,}',
        'money': r'(?:Rs\.?|₹|\$)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        'general': r'\b(\d+(?:\.\d+)?)\b'
    }
    
    @classmethod
    def extract_numbers(cls, text: str) -> Dict[str, List[Any]]:
        """Extract all numerical values from text"""
        extracted = {}
        
        for name, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[name] = matches
        
        return extracted
    
    @classmethod
    def extract_specific_values(
        cls,
        text: str,
        value_type: str
    ) -> List[str]:
        """Extract specific type of numerical values"""
        pattern = cls.PATTERNS.get(value_type, cls.PATTERNS['general'])
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [str(m) if not isinstance(m, tuple) else str(m[0]) for m in matches]


class TemporalExtractor:
    """Extracts temporal information from text"""
    
    DATE_PATTERNS = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
    ]
    
    RELATIVE_PATTERNS = [
        r'(last|next|this)\s+(week|month|year|semester)',
        r'(before|after|during)\s+(\w+)',
        r'(deadline|due|expires?|valid\s+until)',
    ]
    
    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        """Extract date expressions from text"""
        dates = []
        for pattern in cls.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend([''.join(m) if isinstance(m, tuple) else m for m in matches])
        return dates
    
    @classmethod
    def extract_temporal_refs(cls, text: str) -> List[str]:
        """Extract relative temporal references"""
        refs = []
        for pattern in cls.RELATIVE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend([''.join(m) if isinstance(m, tuple) else m for m in matches])
        return refs


class EntityExtractor:
    """Extracts named entities and key facts"""
    
    # Common entity patterns
    PATTERNS = {
        'email': r'[\w.-]+@[\w.-]+\.\w+',
        'phone': r'(?:\+91[-\s]?)?[6-9]\d{9}|\d{2,4}[-\s]?\d{6,8}',
        'course_code': r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?',
        'person_name': r'(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        'room': r'(?:Room|Office|Lab)\s*[A-Z]?\d+[A-Z]?',
    }
    
    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract all entity types from text"""
        entities = {}
        
        for entity_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities
    
    @classmethod
    def extract_key_value_pairs(cls, text: str) -> Dict[str, str]:
        """Extract key-value pairs from structured text"""
        pairs = {}
        
        # Pattern: "Key: Value" or "Key - Value"
        pattern = r'([A-Za-z\s]+):\s*([^\n:]+)'
        matches = re.findall(pattern, text)
        
        for key, value in matches:
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                pairs[key] = value
        
        return pairs


class ConflictDetector:
    """
    Comprehensive conflict detection system for RAG outputs.
    Identifies contradictions, inconsistencies, and conflicts across sources.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        conflict_confidence_threshold: float = 0.6
    ):
        """
        Initialize the conflict detector.
        
        Args:
            similarity_threshold: Minimum similarity to consider texts related
            conflict_confidence_threshold: Minimum confidence to report a conflict
        """
        self.similarity_threshold = similarity_threshold
        self.conflict_confidence_threshold = conflict_confidence_threshold
        self.numerical_extractor = NumericalExtractor()
        self.temporal_extractor = TemporalExtractor()
        self.entity_extractor = EntityExtractor()
        self.conflict_history: List[ConflictReport] = []
        
        # Negation patterns for semantic conflict detection
        self.negation_patterns = [
            (r'\bnot\b', r'\b(?:is|are|was|were|has|have|can|will)\b'),
            (r'\bno\b', r'\b(?:there\s+(?:is|are))\b'),
            (r'\bnever\b', r'\balways\b'),
            (r'\bforbidden\b', r'\ballowed\b'),
            (r'\brequired\b', r'\boptional\b'),
            (r'\bmandatory\b', r'\bvoluntary\b'),
        ]
        
        # Contradiction word pairs
        self.contradiction_pairs = [
            ('increase', 'decrease'),
            ('open', 'closed'),
            ('available', 'unavailable'),
            ('active', 'inactive'),
            ('present', 'absent'),
            ('include', 'exclude'),
            ('approve', 'reject'),
            ('pass', 'fail'),
            ('start', 'end'),
            ('before', 'after'),
        ]
    
    def detect_conflicts(
        self,
        documents: List[str],
        metadatas: List[Dict],
        query: Optional[str] = None
    ) -> ConflictReport:
        """
        Detect conflicts across a set of documents.
        
        Args:
            documents: List of document texts
            metadatas: List of document metadata
            query: Optional query for context
            
        Returns:
            ConflictReport with all detected conflicts
        """
        conflicts = []
        
        # Compare all pairs of documents
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc_conflicts = self._compare_documents(
                    documents[i], metadatas[i],
                    documents[j], metadatas[j]
                )
                conflicts.extend(doc_conflicts)
        
        # Filter by confidence threshold
        significant_conflicts = [
            c for c in conflicts 
            if c.detection_confidence >= self.conflict_confidence_threshold
        ]
        
        # Sort by severity and confidence
        significant_conflicts.sort(
            key=lambda x: (
                list(ConflictSeverity).index(x.severity),
                -x.detection_confidence
            )
        )
        
        # Calculate reliability impact
        reliability_impact = self._calculate_reliability_impact(significant_conflicts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(significant_conflicts)
        
        report = ConflictReport(
            report_id=hashlib.md5(
                f"{query or 'no_query'}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            query=query or "",
            conflicts=significant_conflicts,
            total_sources_analyzed=len(documents),
            conflict_free=len(significant_conflicts) == 0,
            overall_reliability_impact=reliability_impact,
            generated_at=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
        self.conflict_history.append(report)
        
        logger.info(
            f"Conflict detection complete: {len(significant_conflicts)} conflicts found "
            f"in {len(documents)} sources"
        )
        
        return report
    
    def _compare_documents(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Compare two documents for conflicts"""
        conflicts = []
        
        # Check numerical conflicts
        num_conflicts = self._detect_numerical_conflicts(doc1, meta1, doc2, meta2)
        conflicts.extend(num_conflicts)
        
        # Check temporal conflicts
        temp_conflicts = self._detect_temporal_conflicts(doc1, meta1, doc2, meta2)
        conflicts.extend(temp_conflicts)
        
        # Check entity conflicts
        entity_conflicts = self._detect_entity_conflicts(doc1, meta1, doc2, meta2)
        conflicts.extend(entity_conflicts)
        
        # Check semantic conflicts
        semantic_conflicts = self._detect_semantic_conflicts(doc1, meta1, doc2, meta2)
        conflicts.extend(semantic_conflicts)
        
        # Check factual conflicts
        factual_conflicts = self._detect_factual_conflicts(doc1, meta1, doc2, meta2)
        conflicts.extend(factual_conflicts)
        
        return conflicts
    
    def _detect_numerical_conflicts(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Detect conflicting numerical values"""
        conflicts = []
        
        nums1 = self.numerical_extractor.extract_numbers(doc1)
        nums2 = self.numerical_extractor.extract_numbers(doc2)
        
        # Check each numerical category
        for category in set(nums1.keys()) & set(nums2.keys()):
            values1 = set(str(v) for v in nums1[category])
            values2 = set(str(v) for v in nums2[category])
            
            # If same category has different values, potential conflict
            if values1 and values2 and values1 != values2:
                # Check if documents are about the same subject
                if self._are_documents_related(doc1, doc2):
                    conflict = DetectedConflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.NUMERICAL,
                        severity=self._assess_numerical_severity(category, values1, values2),
                        description=f"Conflicting {category} values found: {values1} vs {values2}",
                        evidence_a=ConflictEvidence(
                            source_file=meta1.get('source', 'unknown'),
                            document_id=meta1.get('doc_id', 'unknown'),
                            content=doc1,
                            metadata=meta1
                        ),
                        evidence_b=ConflictEvidence(
                            source_file=meta2.get('source', 'unknown'),
                            document_id=meta2.get('doc_id', 'unknown'),
                            content=doc2,
                            metadata=meta2
                        ),
                        conflicting_claims=(str(values1), str(values2)),
                        detection_confidence=0.75,
                        resolution_suggestion=f"Verify the correct {category} value from official sources"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_temporal_conflicts(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Detect conflicting temporal information"""
        conflicts = []
        
        dates1 = self.temporal_extractor.extract_dates(doc1)
        dates2 = self.temporal_extractor.extract_dates(doc2)
        
        # Check for conflicting dates about the same event
        if dates1 and dates2 and self._are_documents_related(doc1, doc2):
            # Compare dates
            common_context = self._find_common_context(doc1, doc2)
            if common_context:
                if set(dates1) != set(dates2):
                    conflict = DetectedConflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.TEMPORAL,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Conflicting dates found: {dates1} vs {dates2}",
                        evidence_a=ConflictEvidence(
                            source_file=meta1.get('source', 'unknown'),
                            document_id=meta1.get('doc_id', 'unknown'),
                            content=doc1,
                            metadata=meta1,
                            timestamp=meta1.get('timestamp')
                        ),
                        evidence_b=ConflictEvidence(
                            source_file=meta2.get('source', 'unknown'),
                            document_id=meta2.get('doc_id', 'unknown'),
                            content=doc2,
                            metadata=meta2,
                            timestamp=meta2.get('timestamp')
                        ),
                        conflicting_claims=(str(dates1), str(dates2)),
                        detection_confidence=0.7,
                        resolution_suggestion="Check for the most recent update or official announcement"
                    )
                    conflicts.append(conflict)
        
        # Check for outdated information
        outdated = self._check_outdated_info(meta1, meta2)
        if outdated:
            conflicts.append(outdated)
        
        return conflicts
    
    def _detect_entity_conflicts(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Detect conflicting entity information"""
        conflicts = []
        
        entities1 = self.entity_extractor.extract_entities(doc1)
        entities2 = self.entity_extractor.extract_entities(doc2)
        kvs1 = self.entity_extractor.extract_key_value_pairs(doc1)
        kvs2 = self.entity_extractor.extract_key_value_pairs(doc2)
        
        # Check key-value conflicts
        common_keys = set(kvs1.keys()) & set(kvs2.keys())
        for key in common_keys:
            if kvs1[key].lower() != kvs2[key].lower():
                conflict = DetectedConflict(
                    conflict_id=self._generate_conflict_id(),
                    conflict_type=ConflictType.ATTRIBUTION,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Conflicting values for '{key}': '{kvs1[key]}' vs '{kvs2[key]}'",
                    evidence_a=ConflictEvidence(
                        source_file=meta1.get('source', 'unknown'),
                        document_id=meta1.get('doc_id', 'unknown'),
                        content=doc1,
                        metadata=meta1
                    ),
                    evidence_b=ConflictEvidence(
                        source_file=meta2.get('source', 'unknown'),
                        document_id=meta2.get('doc_id', 'unknown'),
                        content=doc2,
                        metadata=meta2
                    ),
                    conflicting_claims=(f"{key}: {kvs1[key]}", f"{key}: {kvs2[key]}"),
                    detection_confidence=0.8,
                    resolution_suggestion=f"Verify the correct '{key}' value"
                )
                conflicts.append(conflict)
        
        # Check email conflicts for same person/course
        if 'email' in entities1 and 'email' in entities2:
            if entities1['email'] != entities2['email'] and self._are_documents_related(doc1, doc2):
                conflicts.append(self._create_entity_conflict(
                    'email', entities1['email'], entities2['email'],
                    meta1, meta2, doc1, doc2
                ))
        
        return conflicts
    
    def _detect_semantic_conflicts(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Detect semantically contradictory statements"""
        conflicts = []
        
        doc1_lower = doc1.lower()
        doc2_lower = doc2.lower()
        
        # Check for contradiction pairs
        for word1, word2 in self.contradiction_pairs:
            if word1 in doc1_lower and word2 in doc2_lower:
                if self._are_documents_related(doc1, doc2):
                    conflicts.append(self._create_semantic_conflict(
                        word1, word2, meta1, meta2, doc1, doc2
                    ))
            elif word2 in doc1_lower and word1 in doc2_lower:
                if self._are_documents_related(doc1, doc2):
                    conflicts.append(self._create_semantic_conflict(
                        word2, word1, meta1, meta2, doc1, doc2
                    ))
        
        # Check for negation patterns
        for pos_pattern, neg_context in self.negation_patterns:
            has_positive_1 = re.search(pos_pattern, doc1_lower) and re.search(neg_context, doc1_lower)
            has_positive_2 = re.search(pos_pattern, doc2_lower) and re.search(neg_context, doc2_lower)
            
            if has_positive_1 != has_positive_2 and self._are_documents_related(doc1, doc2):
                conflicts.append(DetectedConflict(
                    conflict_id=self._generate_conflict_id(),
                    conflict_type=ConflictType.SEMANTIC,
                    severity=ConflictSeverity.MEDIUM,
                    description="Potential semantic contradiction detected (negation pattern)",
                    evidence_a=ConflictEvidence(
                        source_file=meta1.get('source', 'unknown'),
                        document_id=meta1.get('doc_id', 'unknown'),
                        content=doc1,
                        metadata=meta1
                    ),
                    evidence_b=ConflictEvidence(
                        source_file=meta2.get('source', 'unknown'),
                        document_id=meta2.get('doc_id', 'unknown'),
                        content=doc2,
                        metadata=meta2
                    ),
                    conflicting_claims=(
                        doc1[:100] + "...",
                        doc2[:100] + "..."
                    ),
                    detection_confidence=0.65,
                    resolution_suggestion="Review both sources to determine the accurate statement"
                ))
        
        return conflicts
    
    def _detect_factual_conflicts(
        self,
        doc1: str,
        meta1: Dict,
        doc2: str,
        meta2: Dict
    ) -> List[DetectedConflict]:
        """Detect direct factual contradictions"""
        conflicts = []
        
        # Extract course codes and check for conflicting info
        codes1 = re.findall(r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?', doc1)
        codes2 = re.findall(r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?', doc2)
        
        common_codes = set(codes1) & set(codes2)
        
        if common_codes:
            # Same course mentioned - check for conflicting details
            kvs1 = self.entity_extractor.extract_key_value_pairs(doc1)
            kvs2 = self.entity_extractor.extract_key_value_pairs(doc2)
            
            critical_keys = ['instructor', 'credits', 'prerequisites', 'semester']
            for key in critical_keys:
                if key in kvs1 and key in kvs2:
                    if kvs1[key].lower() != kvs2[key].lower():
                        conflict = DetectedConflict(
                            conflict_id=self._generate_conflict_id(),
                            conflict_type=ConflictType.FACTUAL,
                            severity=ConflictSeverity.HIGH,
                            description=f"Conflicting {key} for course(s) {common_codes}",
                            evidence_a=ConflictEvidence(
                                source_file=meta1.get('source', 'unknown'),
                                document_id=meta1.get('doc_id', 'unknown'),
                                content=doc1,
                                metadata=meta1
                            ),
                            evidence_b=ConflictEvidence(
                                source_file=meta2.get('source', 'unknown'),
                                document_id=meta2.get('doc_id', 'unknown'),
                                content=doc2,
                                metadata=meta2
                            ),
                            conflicting_claims=(
                                f"{key}: {kvs1[key]}",
                                f"{key}: {kvs2[key]}"
                            ),
                            detection_confidence=0.85,
                            resolution_suggestion=f"Verify {key} from official course catalog"
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _are_documents_related(self, doc1: str, doc2: str) -> bool:
        """Check if two documents are about related topics"""
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'to',
                    'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'and',
                    'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down'}
        
        words1 -= stopwords
        words2 -= stopwords
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        min_size = min(len(words1), len(words2))
        
        similarity = intersection / min_size if min_size > 0 else 0
        return similarity >= self.similarity_threshold
    
    def _find_common_context(self, doc1: str, doc2: str) -> Optional[str]:
        """Find common context between two documents"""
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        common = words1 & words2
        
        # Filter meaningful words
        meaningful = [w for w in common if len(w) > 3]
        
        if meaningful:
            return ' '.join(meaningful[:10])
        return None
    
    def _check_outdated_info(
        self,
        meta1: Dict,
        meta2: Dict
    ) -> Optional[DetectedConflict]:
        """Check if one source might be outdated compared to another"""
        ts1 = meta1.get('timestamp')
        ts2 = meta2.get('timestamp')
        
        if ts1 and ts2:
            try:
                dt1 = datetime.fromisoformat(ts1)
                dt2 = datetime.fromisoformat(ts2)
                
                diff_days = abs((dt1 - dt2).days)
                
                if diff_days > 180:  # 6 months difference
                    older = meta1 if dt1 < dt2 else meta2
                    newer = meta2 if dt1 < dt2 else meta1
                    
                    return DetectedConflict(
                        conflict_id=self._generate_conflict_id(),
                        conflict_type=ConflictType.OUTDATED,
                        severity=ConflictSeverity.LOW,
                        description=f"Source time difference of {diff_days} days detected",
                        evidence_a=ConflictEvidence(
                            source_file=older.get('source', 'unknown'),
                            document_id=older.get('doc_id', 'unknown'),
                            content="Older source",
                            metadata=older,
                            timestamp=older.get('timestamp')
                        ),
                        evidence_b=ConflictEvidence(
                            source_file=newer.get('source', 'unknown'),
                            document_id=newer.get('doc_id', 'unknown'),
                            content="Newer source",
                            metadata=newer,
                            timestamp=newer.get('timestamp')
                        ),
                        conflicting_claims=(
                            f"Updated: {older.get('timestamp', 'unknown')}",
                            f"Updated: {newer.get('timestamp', 'unknown')}"
                        ),
                        detection_confidence=0.6,
                        resolution_suggestion="Prefer information from the more recent source"
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _generate_conflict_id(self) -> str:
        """Generate unique conflict ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}_{id(self)}".encode()
        ).hexdigest()[:10]
    
    def _assess_numerical_severity(
        self,
        category: str,
        values1: Set[str],
        values2: Set[str]
    ) -> ConflictSeverity:
        """Assess severity of numerical conflict"""
        critical_categories = ['credits', 'percentage']
        high_categories = ['count', 'money']
        
        if category in critical_categories:
            return ConflictSeverity.HIGH
        elif category in high_categories:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW
    
    def _create_entity_conflict(
        self,
        entity_type: str,
        values1: List[str],
        values2: List[str],
        meta1: Dict,
        meta2: Dict,
        doc1: str,
        doc2: str
    ) -> DetectedConflict:
        """Create an entity conflict"""
        return DetectedConflict(
            conflict_id=self._generate_conflict_id(),
            conflict_type=ConflictType.ATTRIBUTION,
            severity=ConflictSeverity.MEDIUM,
            description=f"Conflicting {entity_type} values",
            evidence_a=ConflictEvidence(
                source_file=meta1.get('source', 'unknown'),
                document_id=meta1.get('doc_id', 'unknown'),
                content=doc1,
                metadata=meta1
            ),
            evidence_b=ConflictEvidence(
                source_file=meta2.get('source', 'unknown'),
                document_id=meta2.get('doc_id', 'unknown'),
                content=doc2,
                metadata=meta2
            ),
            conflicting_claims=(str(values1), str(values2)),
            detection_confidence=0.7,
            resolution_suggestion=f"Verify the correct {entity_type}"
        )
    
    def _create_semantic_conflict(
        self,
        word1: str,
        word2: str,
        meta1: Dict,
        meta2: Dict,
        doc1: str,
        doc2: str
    ) -> DetectedConflict:
        """Create a semantic conflict"""
        return DetectedConflict(
            conflict_id=self._generate_conflict_id(),
            conflict_type=ConflictType.SEMANTIC,
            severity=ConflictSeverity.MEDIUM,
            description=f"Contradictory terms found: '{word1}' vs '{word2}'",
            evidence_a=ConflictEvidence(
                source_file=meta1.get('source', 'unknown'),
                document_id=meta1.get('doc_id', 'unknown'),
                content=doc1,
                metadata=meta1
            ),
            evidence_b=ConflictEvidence(
                source_file=meta2.get('source', 'unknown'),
                document_id=meta2.get('doc_id', 'unknown'),
                content=doc2,
                metadata=meta2
            ),
            conflicting_claims=(
                f"States '{word1}'",
                f"States '{word2}'"
            ),
            detection_confidence=0.65,
            resolution_suggestion="Review context to determine accurate information"
        )
    
    def _calculate_reliability_impact(
        self,
        conflicts: List[DetectedConflict]
    ) -> float:
        """Calculate overall reliability impact of conflicts"""
        if not conflicts:
            return 0.0
        
        severity_weights = {
            ConflictSeverity.CRITICAL: 0.4,
            ConflictSeverity.HIGH: 0.25,
            ConflictSeverity.MEDIUM: 0.15,
            ConflictSeverity.LOW: 0.08,
            ConflictSeverity.INFO: 0.02
        }
        
        total_impact = sum(
            severity_weights.get(c.severity, 0.1) * c.detection_confidence
            for c in conflicts
        )
        
        return min(total_impact, 1.0)
    
    def _generate_recommendations(
        self,
        conflicts: List[DetectedConflict]
    ) -> List[str]:
        """Generate recommendations based on detected conflicts"""
        recommendations = []
        
        if not conflicts:
            recommendations.append("No conflicts detected - information appears consistent")
            return recommendations
        
        # Count conflict types
        type_counts = defaultdict(int)
        for conflict in conflicts:
            type_counts[conflict.conflict_type] += 1
        
        # Generate type-specific recommendations
        if type_counts[ConflictType.FACTUAL] > 0:
            recommendations.append(
                "⚠️ Factual conflicts detected - verify information from official sources"
            )
        
        if type_counts[ConflictType.TEMPORAL] > 0:
            recommendations.append(
                "📅 Temporal conflicts found - check for the most recent updates"
            )
        
        if type_counts[ConflictType.NUMERICAL] > 0:
            recommendations.append(
                "🔢 Numerical discrepancies found - verify specific numbers carefully"
            )
        
        if type_counts[ConflictType.OUTDATED] > 0:
            recommendations.append(
                "🕐 Some sources may be outdated - prefer recent information"
            )
        
        # Critical conflicts
        critical_count = sum(
            1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL
        )
        if critical_count > 0:
            recommendations.insert(
                0,
                f"🚨 {critical_count} critical conflict(s) require immediate attention"
            )
        
        return recommendations
    
    def format_conflict_report(
        self,
        report: ConflictReport,
        format_type: str = "detailed"
    ) -> str:
        """
        Format conflict report for display.
        
        Args:
            report: Conflict report to format
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
    
    def _format_detailed(self, report: ConflictReport) -> str:
        """Format detailed conflict report"""
        lines = [
            "═" * 60,
            "📋 **CONFLICT DETECTION REPORT**",
            "═" * 60,
            f"Report ID: {report.report_id}",
            f"Query: {report.query[:50]}..." if len(report.query) > 50 else f"Query: {report.query}",
            f"Sources Analyzed: {report.total_sources_analyzed}",
            f"Conflicts Found: {len(report.conflicts)}",
            f"Reliability Impact: {report.overall_reliability_impact:.1%}",
            f"Status: {'✅ Conflict-Free' if report.conflict_free else '⚠️ Conflicts Detected'}",
            "",
        ]
        
        if report.conflicts:
            lines.append("**Detected Conflicts:**")
            lines.append("-" * 40)
            
            for i, conflict in enumerate(report.conflicts, 1):
                severity_emoji = {
                    ConflictSeverity.CRITICAL: "🚨",
                    ConflictSeverity.HIGH: "🔴",
                    ConflictSeverity.MEDIUM: "🟡",
                    ConflictSeverity.LOW: "🟢",
                    ConflictSeverity.INFO: "ℹ️"
                }
                
                lines.extend([
                    f"\n**Conflict #{i}** {severity_emoji.get(conflict.severity, '')}",
                    f"  Type: {conflict.conflict_type.value.title()}",
                    f"  Severity: {conflict.severity.value.title()}",
                    f"  Description: {conflict.description}",
                    f"  Source A: {conflict.evidence_a.source_file}",
                    f"  Source B: {conflict.evidence_b.source_file}",
                    f"  Confidence: {conflict.detection_confidence:.1%}",
                    f"  Suggestion: {conflict.resolution_suggestion}",
                ])
        
        lines.extend([
            "",
            "**Recommendations:**"
        ])
        for rec in report.recommendations:
            lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def _format_summary(self, report: ConflictReport) -> str:
        """Format summary conflict report"""
        status = "✅ No conflicts" if report.conflict_free else f"⚠️ {len(report.conflicts)} conflicts"
        return (
            f"Conflict Report: {status} | "
            f"Sources: {report.total_sources_analyzed} | "
            f"Impact: {report.overall_reliability_impact:.1%}"
        )
    
    def _format_compact(self, report: ConflictReport) -> str:
        """Format compact conflict report"""
        if report.conflict_free:
            return "✅ No conflicts detected"
        
        lines = [f"⚠️ {len(report.conflicts)} conflict(s) detected:"]
        for conflict in report.conflicts[:3]:  # Show top 3
            lines.append(
                f"  • {conflict.conflict_type.value}: {conflict.description[:50]}..."
            )
        
        if len(report.conflicts) > 3:
            lines.append(f"  ... and {len(report.conflicts) - 3} more")
        
        return "\n".join(lines)
    
    def get_conflict_history(self, limit: int = 10) -> List[ConflictReport]:
        """Get recent conflict reports"""
        return self.conflict_history[-limit:]
    
    def export_report(self, report: ConflictReport) -> str:
        """Export report as JSON"""
        return json.dumps(report.to_dict(), indent=2)


# Global instance
_conflict_detector = None

def get_conflict_detector() -> ConflictDetector:
    """Get or create conflict detector instance"""
    global _conflict_detector
    if _conflict_detector is None:
        _conflict_detector = ConflictDetector()
    return _conflict_detector


def detect_conflicts(
    documents: List[str],
    metadatas: List[Dict],
    query: Optional[str] = None
) -> ConflictReport:
    """
    Convenience function to detect conflicts.
    
    Args:
        documents: List of document texts
        metadatas: Document metadata
        query: Optional query for context
        
    Returns:
        ConflictReport
    """
    detector = get_conflict_detector()
    return detector.detect_conflicts(documents, metadatas, query)
