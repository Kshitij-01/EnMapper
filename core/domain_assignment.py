"""
Domain Assignment Engine for EnMapper

This module implements the core domain assignment logic using:
- RAG-based candidate generation
- Multi-factor scoring (name_sim + regex_strength + value_sim + unit_compat)
- Confidence banding and arbitration
- Evidence collection and persistence

Phase 2 Component: LLM-D (Domains) specialist
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None

from .domain_catalog import (
    DomainCatalog, DomainDefinition, Evidence, DomainAssignment, 
    ConfidenceBand, get_domain_catalog
)

logger = logging.getLogger(__name__)


@dataclass
class DomainScoringWeights:
    """Configurable weights for domain scoring algorithm."""
    alpha: float = 0.4    # Name similarity weight
    beta: float = 0.3     # Regex strength weight  
    gamma: float = 0.2    # Value similarity weight
    epsilon: float = 0.1  # Unit compatibility weight
    
    def __post_init__(self):
        """Ensure weights sum to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.epsilon
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Scoring weights sum to {total}, normalizing to 1.0")
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.epsilon /= total


@dataclass 
class DomainThresholds:
    """Confidence thresholds for domain assignment."""
    tau_high: float = 0.82  # ≥ auto-assign (High confidence)
    tau_low: float = 0.55   # < becomes unknown (Low confidence)
    
    def get_confidence_band(self, score: float) -> ConfidenceBand:
        """Determine confidence band from score."""
        if score >= self.tau_high:
            return ConfidenceBand.HIGH
        elif score >= self.tau_low:
            return ConfidenceBand.BORDERLINE
        else:
            return ConfidenceBand.LOW


@dataclass
class ColumnInfo:
    """Information about a column for domain assignment."""
    name: str
    sample_values: List[str] = field(default_factory=list)
    data_type: str = "unknown"
    null_count: int = 0
    total_count: int = 0
    unique_count: int = 0
    
    def get_header_tokens(self) -> List[str]:
        """Extract tokens from column name."""
        # Split on common delimiters and convert to lowercase
        tokens = re.split(r'[_\-\s.]+', self.name.lower())
        return [t for t in tokens if t and len(t) > 1]


class DomainAssignmentEngine:
    """Core engine for domain assignment using RAG and scoring."""
    
    def __init__(self, 
                 catalog: Optional[DomainCatalog] = None,
                 weights: Optional[DomainScoringWeights] = None,
                 thresholds: Optional[DomainThresholds] = None):
        
        self.catalog = catalog or get_domain_catalog()
        self.weights = weights or DomainScoringWeights()
        self.thresholds = thresholds or DomainThresholds()
        
        # Cache for performance
        self._embedding_cache = {}
        
        logger.info(f"🧠 Domain Assignment Engine initialized")
        logger.info(f"📊 Weights: α={self.weights.alpha:.2f}, β={self.weights.beta:.2f}, "
                   f"γ={self.weights.gamma:.2f}, ε={self.weights.epsilon:.2f}")
        logger.info(f"🎯 Thresholds: τ_high={self.thresholds.tau_high}, τ_low={self.thresholds.tau_low}")
    
    def assign_domains(self, columns: List[ColumnInfo], run_id: str = "") -> List[DomainAssignment]:
        """Assign domains to a list of columns."""
        assignments = []
        
        for column in columns:
            assignment = self.assign_domain(column, run_id)
            assignments.append(assignment)
            
            logger.debug(f"Column '{column.name}' → {assignment.domain_name or 'UNKNOWN'} "
                        f"({assignment.confidence_band.value}, {assignment.confidence_score:.3f})")
        
        return assignments
    
    def assign_domain(self, column: ColumnInfo, run_id: str = "") -> DomainAssignment:
        """Assign a domain to a single column."""
        
        # Step 1: Generate candidate domains using RAG
        candidates = self._generate_candidates(column)
        
        # Step 2: Score each candidate
        scored_candidates = []
        for domain in candidates:
            evidence = self._extract_evidence(column, domain)
            score = self._calculate_score(evidence)
            evidence.composite_score = score
            evidence.confidence_band = self.thresholds.get_confidence_band(score)
            
            scored_candidates.append((domain, evidence, score))
        
        # Step 3: Select best candidate
        if scored_candidates:
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            best_domain, best_evidence, best_score = scored_candidates[0]
            
            return DomainAssignment(
                column_name=column.name,
                domain_id=best_domain.domain_id,
                domain_name=best_domain.name,
                confidence_score=best_score,
                confidence_band=best_evidence.confidence_band,
                evidence=best_evidence,
                run_id=run_id
            )
        else:
            # No candidates found
            return DomainAssignment(
                column_name=column.name,
                domain_id=None,
                domain_name=None,
                confidence_score=0.0,
                confidence_band=ConfidenceBand.LOW,
                evidence=Evidence(column_name=column.name),
                run_id=run_id
            )
    
    def _generate_candidates(self, column: ColumnInfo, max_candidates: int = 10) -> List[DomainDefinition]:
        """Generate candidate domains using RAG search."""
        
        # Build search query from column name and context
        search_terms = [column.name]
        search_terms.extend(column.get_header_tokens())
        
        # Add data type hints
        if column.data_type:
            search_terms.append(column.data_type)
        
        search_query = " ".join(search_terms)
        
        # Search using catalog's RAG capabilities
        search_results = self.catalog.search_domains(search_query, limit=max_candidates)
        
        # Extract domains from results
        candidates = [domain for domain, similarity in search_results]
        
        # Also include exact name matches
        exact_matches = self._find_exact_matches(column)
        for domain in exact_matches:
            if domain not in candidates:
                candidates.insert(0, domain)  # Prioritize exact matches
        
        return candidates[:max_candidates]
    
    def _find_exact_matches(self, column: ColumnInfo) -> List[DomainDefinition]:
        """Find domains with exact name or alias matches."""
        matches = []
        column_name_lower = column.name.lower()
        
        for domain in self.catalog.domains.values():
            # Check exact name match
            if domain.name.lower() == column_name_lower:
                matches.append(domain)
                continue
            
            # Check alias matches
            for alias in domain.aliases:
                if alias.lower() == column_name_lower:
                    matches.append(domain)
                    break
        
        return matches
    
    def _extract_evidence(self, column: ColumnInfo, domain: DomainDefinition) -> Evidence:
        """Extract evidence for domain assignment."""
        evidence = Evidence(
            column_name=column.name,
            sample_values=column.sample_values[:10]  # Limit for performance
        )
        
        # Calculate name similarity
        evidence.name_similarity = self._calculate_name_similarity(column, domain)
        evidence.matching_aliases = self._find_matching_aliases(column, domain)
        
        # Calculate regex strength
        evidence.regex_strength = self._calculate_regex_strength(column, domain)
        evidence.matching_patterns = self._find_matching_patterns(column, domain)
        
        # Calculate value similarity (if embeddings available and domain has examples)
        evidence.value_similarity = self._calculate_value_similarity(column, domain)
        
        # Calculate unit compatibility
        evidence.unit_compatibility = self._calculate_unit_compatibility(column, domain)
        evidence.matching_units = self._find_matching_units(column, domain)
        
        # Extract header tokens
        evidence.header_tokens = column.get_header_tokens()
        
        return evidence
    
    def _calculate_score(self, evidence: Evidence) -> float:
        """Calculate composite domain assignment score."""
        score = (
            self.weights.alpha * evidence.name_similarity +
            self.weights.beta * evidence.regex_strength +
            self.weights.gamma * evidence.value_similarity +
            self.weights.epsilon * evidence.unit_compatibility
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_name_similarity(self, column: ColumnInfo, domain: DomainDefinition) -> float:
        """Calculate name similarity between column and domain."""
        column_name = column.name.lower()
        
        # Exact match with domain name
        if column_name == domain.name.lower():
            return 1.0
        
        # Exact match with any alias
        for alias in domain.aliases:
            if column_name == alias.lower():
                return 0.9
        
        # Partial matches using token overlap
        column_tokens = set(column.get_header_tokens())
        
        # Check domain name tokens
        domain_tokens = set(re.split(r'[._\-\s]+', domain.name.lower()))
        name_overlap = len(column_tokens & domain_tokens) / max(len(column_tokens | domain_tokens), 1)
        
        # Check header tokens
        header_tokens = set(token.lower() for token in domain.header_tokens)
        header_overlap = len(column_tokens & header_tokens) / max(len(column_tokens), 1)
        
        # Return best overlap score
        return max(name_overlap * 0.8, header_overlap * 0.7)
    
    def _find_matching_aliases(self, column: ColumnInfo, domain: DomainDefinition) -> List[str]:
        """Find aliases that match the column name."""
        matches = []
        column_name_lower = column.name.lower()
        
        for alias in domain.aliases:
            if alias.lower() in column_name_lower or column_name_lower in alias.lower():
                matches.append(alias)
        
        return matches
    
    def _calculate_regex_strength(self, column: ColumnInfo, domain: DomainDefinition) -> float:
        """Calculate how well sample values match domain regex patterns."""
        if not domain.regex_patterns or not column.sample_values:
            return 0.0
        
        total_matches = 0
        total_values = len(column.sample_values)
        
        for pattern_str in domain.regex_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = sum(1 for value in column.sample_values if pattern.match(str(value)))
                match_ratio = matches / total_values
                total_matches = max(total_matches, match_ratio)  # Take best pattern match
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern_str}")
                continue
        
        return total_matches
    
    def _find_matching_patterns(self, column: ColumnInfo, domain: DomainDefinition) -> List[str]:
        """Find regex patterns that match sample values."""
        matching = []
        
        for pattern_str in domain.regex_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if any(pattern.match(str(value)) for value in column.sample_values):
                    matching.append(pattern_str)
            except re.error:
                continue
        
        return matching
    
    def _calculate_value_similarity(self, column: ColumnInfo, domain: DomainDefinition) -> float:
        """Calculate value similarity using embeddings."""
        if not self.catalog.enable_embeddings or not column.sample_values:
            return 0.0
        
        if domain.value_embedding is None or not hasattr(domain, 'value_embedding'):
            return 0.0
        
        try:
            # Get embeddings for sample values
            sample_embeddings = self.catalog.embedding_model.encode(column.sample_values[:5])  # Limit for performance
            column_centroid = np.mean(sample_embeddings, axis=0)
            
            # Calculate cosine similarity with domain centroid
            similarity = np.dot(column_centroid, domain.value_embedding) / (
                np.linalg.norm(column_centroid) * np.linalg.norm(domain.value_embedding)
            )
            
            return max(0.0, float(similarity))  # Ensure non-negative
            
        except Exception as e:
            logger.debug(f"Value similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_unit_compatibility(self, column: ColumnInfo, domain: DomainDefinition) -> float:
        """Calculate unit compatibility score."""
        if not domain.unit_cues:
            return 0.0
        
        # Check column name for unit indicators
        column_text = column.name.lower()
        
        matches = 0
        for unit_cue in domain.unit_cues:
            if unit_cue.lower() in column_text:
                matches += 1
        
        # Also check sample values for unit indicators
        sample_text = " ".join(str(v) for v in column.sample_values[:5]).lower()
        for unit_cue in domain.unit_cues:
            if unit_cue.lower() in sample_text:
                matches += 1
                break  # Avoid double counting
        
        return min(matches / len(domain.unit_cues), 1.0)
    
    def _find_matching_units(self, column: ColumnInfo, domain: DomainDefinition) -> List[str]:
        """Find unit cues that match the column."""
        matches = []
        column_text = (column.name + " " + " ".join(str(v) for v in column.sample_values[:5])).lower()
        
        for unit_cue in domain.unit_cues:
            if unit_cue.lower() in column_text:
                matches.append(unit_cue)
        
        return matches
    
    def get_assignment_summary(self, assignments: List[DomainAssignment]) -> Dict[str, Any]:
        """Get summary statistics for domain assignments."""
        if not assignments:
            return {"total": 0}
        
        confidence_counts = {band.value: 0 for band in ConfidenceBand}
        assigned_count = 0
        total_score = 0.0
        
        for assignment in assignments:
            confidence_counts[assignment.confidence_band.value] += 1
            total_score += assignment.confidence_score
            
            if assignment.domain_id:
                assigned_count += 1
        
        return {
            "total": len(assignments),
            "assigned": assigned_count,
            "unassigned": len(assignments) - assigned_count,
            "confidence_distribution": confidence_counts,
            "average_score": total_score / len(assignments),
            "assignment_rate": assigned_count / len(assignments)
        }


# Convenience functions
def assign_domains_to_columns(columns: List[ColumnInfo], 
                            run_id: str = "",
                            weights: Optional[DomainScoringWeights] = None,
                            thresholds: Optional[DomainThresholds] = None) -> List[DomainAssignment]:
    """Convenience function to assign domains to columns."""
    engine = DomainAssignmentEngine(weights=weights, thresholds=thresholds)
    return engine.assign_domains(columns, run_id)


def assign_domain_to_column(column: ColumnInfo,
                          run_id: str = "",
                          weights: Optional[DomainScoringWeights] = None,
                          thresholds: Optional[DomainThresholds] = None) -> DomainAssignment:
    """Convenience function to assign domain to a single column."""
    engine = DomainAssignmentEngine(weights=weights, thresholds=thresholds)
    return engine.assign_domain(column, run_id)


