"""
Domain Catalog System for EnMapper

This module implements the Domain Catalog with RAG (Retrieval-Augmented Generation)
capabilities for semantic domain detection and assignment.

Phase 2 Components:
- Domain definitions with examples, aliases, regex patterns
- Vector embeddings for semantic similarity search
- Staging workflow for catalog updates
- Evidence extraction and scoring
"""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None
    SentenceTransformer = None

# Safe ndarray type alias that doesn't break when NumPy is unavailable
NDArrayType = Any if np is None else np.ndarray

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    """Domain classification types."""
    ATOMIC = "atomic"      # Simple semantic type (e.g., email, phone)
    COMPOSITE = "composite"  # Complex type (e.g., address, person_name)


class ConfidenceBand(str, Enum):
    """Domain assignment confidence levels."""
    HIGH = "high"         # τ ≥ 0.82 - auto-assign
    BORDERLINE = "borderline"  # 0.55 ≤ τ < 0.82 - human review
    LOW = "low"           # τ < 0.55 - mark as unknown


@dataclass
class DomainExample:
    """Example value for a domain with metadata."""
    value: str
    context: str = ""  # Optional context description
    is_positive: bool = True  # True for positive examples, False for negatives
    source: str = "curated"  # Source of the example
    
    
@dataclass
class DomainDefinition:
    """Core domain definition with patterns and examples."""
    
    # Identity
    domain_id: str
    name: str  # e.g., "person.email", "contact.phone"
    description: str
    domain_type: DomainType = DomainType.ATOMIC
    
    # Patterns and cues
    aliases: List[str] = field(default_factory=list)  # Alternative names
    regex_patterns: List[str] = field(default_factory=list)  # Value patterns
    unit_cues: List[str] = field(default_factory=list)  # Unit indicators
    header_tokens: List[str] = field(default_factory=list)  # Header keywords
    
    # Examples for training and validation
    positive_examples: List[DomainExample] = field(default_factory=list)
    negative_examples: List[DomainExample] = field(default_factory=list)
    
    # Vector embeddings (computed)
    name_embedding: Optional[NDArrayType] = None
    value_embedding: Optional[NDArrayType] = None  # Centroid of examples
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"
    provenance: str = "curated"  # curated, learned, user_feedback
    
    def add_example(self, value: str, is_positive: bool = True, context: str = "", source: str = "user"):
        """Add a new example to the domain."""
        example = DomainExample(value=value, is_positive=is_positive, context=context, source=source)
        
        if is_positive:
            self.positive_examples.append(example)
        else:
            self.negative_examples.append(example)
            
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_all_name_variants(self) -> List[str]:
        """Get all name variants including aliases."""
        return [self.name] + self.aliases
    
    def get_positive_values(self) -> List[str]:
        """Get all positive example values."""
        return [ex.value for ex in self.positive_examples]


@dataclass
class Evidence:
    """Evidence supporting a domain assignment."""
    
    # Source information
    column_name: str
    sample_values: List[str] = field(default_factory=list)
    
    # Evidence scores (0.0 to 1.0)
    name_similarity: float = 0.0
    regex_strength: float = 0.0
    value_similarity: float = 0.0
    unit_compatibility: float = 0.0
    
    # Supporting details
    matching_aliases: List[str] = field(default_factory=list)
    matching_patterns: List[str] = field(default_factory=list)
    matching_units: List[str] = field(default_factory=list)
    header_tokens: List[str] = field(default_factory=list)
    
    # Confidence calculation
    composite_score: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.LOW


@dataclass
class DomainAssignment:
    """Domain assignment result for a column."""
    
    column_name: str
    domain_id: Optional[str]
    domain_name: Optional[str]
    confidence_score: float
    confidence_band: ConfidenceBand
    evidence: Evidence
    
    # Metadata
    assigned_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    run_id: str = ""
    human_reviewed: bool = False
    human_decision: Optional[str] = None  # approved, rejected, unknown


class DomainCatalog:
    """Domain catalog with RAG capabilities."""
    
    def __init__(self, enable_embeddings: bool = True):
        self.domains: Dict[str, DomainDefinition] = {}
        self.staged_domains: Dict[str, DomainDefinition] = {}
        self.enable_embeddings = enable_embeddings and EMBEDDINGS_AVAILABLE
        self.embedding_model = None
        
        if self.enable_embeddings:
            self._initialize_embeddings()
        
        self._initialize_default_domains()
        
    def _initialize_embeddings(self):
        """Initialize the sentence transformer model."""
        try:
            # Use a lightweight, fast model for production
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model initialized: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            self.enable_embeddings = False
    
    def _initialize_default_domains(self):
        """Initialize with curated domain definitions."""
        
        # Common person domains
        self.add_domain(DomainDefinition(
            domain_id="person.email",
            name="person.email",
            description="Email address for a person",
            aliases=["email", "email_address", "e_mail", "mail"],
            regex_patterns=[
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            ],
            header_tokens=["email", "mail", "e_mail", "contact"],
            positive_examples=[
                DomainExample("john.doe@example.com", "Standard email format"),
                DomainExample("user123@gmail.com", "Gmail address"),
                DomainExample("jane+work@company.org", "Plus addressing"),
                DomainExample("test@subdomain.example.co.uk", "Subdomain email"),
            ]
        ))
        
        self.add_domain(DomainDefinition(
            domain_id="person.phone",
            name="person.phone", 
            description="Phone number for a person",
            aliases=["phone", "telephone", "mobile", "cell", "contact_number"],
            regex_patterns=[
                r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
                r'^\+?[1-9]\d{1,14}$'  # International format
            ],
            header_tokens=["phone", "tel", "mobile", "cell", "contact"],
            positive_examples=[
                DomainExample("(555) 123-4567", "US format with parentheses"),
                DomainExample("555-123-4567", "US format with dashes"),
                DomainExample("+1 555 123 4567", "International US format"),
                DomainExample("5551234567", "Digits only"),
            ]
        ))
        
        self.add_domain(DomainDefinition(
            domain_id="person.name.first",
            name="person.name.first",
            description="First name or given name of a person",
            aliases=["first_name", "given_name", "fname", "forename"],
            header_tokens=["first", "given", "fname", "forename", "name"],
            positive_examples=[
                DomainExample("John", "Common first name"),
                DomainExample("Maria", "International name"),
                DomainExample("Li", "Short name"),
                DomainExample("Jean-Pierre", "Hyphenated name"),
            ]
        ))
        
        self.add_domain(DomainDefinition(
            domain_id="person.name.last",
            name="person.name.last",
            description="Last name or surname of a person",
            aliases=["last_name", "surname", "family_name", "lname"],
            header_tokens=["last", "surname", "family", "lname"],
            positive_examples=[
                DomainExample("Smith", "Common surname"),
                DomainExample("García", "International surname"),
                DomainExample("van der Berg", "Multi-part surname"),
                DomainExample("O'Connor", "Irish surname"),
            ]
        ))
        
        # Financial domains
        self.add_domain(DomainDefinition(
            domain_id="finance.amount",
            name="finance.amount",
            description="Monetary amount or currency value",
            aliases=["amount", "price", "cost", "value", "total"],
            regex_patterns=[
                r'^\$?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{2})?$',
                r'^[0-9]+\.[0-9]{2}$'
            ],
            unit_cues=["$", "USD", "EUR", "currency", "dollars"],
            header_tokens=["amount", "price", "cost", "value", "total", "money"],
            positive_examples=[
                DomainExample("$1,234.56", "US currency format"),
                DomainExample("999.99", "Decimal format"),
                DomainExample("1000", "Integer amount"),
                DomainExample("$0.99", "Small amount"),
            ]
        ))
        
        # Date/time domains
        self.add_domain(DomainDefinition(
            domain_id="temporal.date",
            name="temporal.date",
            description="Date value in various formats",
            aliases=["date", "created_date", "updated_date", "timestamp"],
            regex_patterns=[
                r'^\d{4}-\d{2}-\d{2}$',  # ISO date
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # US format
                r'^\d{1,2}-\d{1,2}-\d{4}$'   # Alternative format
            ],
            header_tokens=["date", "created", "updated", "time", "timestamp"],
            positive_examples=[
                DomainExample("2023-12-25", "ISO date format"),
                DomainExample("12/25/2023", "US date format"),
                DomainExample("25-12-2023", "European format"),
                DomainExample("2023-01-01", "New Year date"),
            ]
        ))
        
        # Identifier domains
        self.add_domain(DomainDefinition(
            domain_id="identifier.uuid",
            name="identifier.uuid",
            description="Universally Unique Identifier",
            aliases=["uuid", "guid", "id", "unique_id"],
            regex_patterns=[
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            ],
            header_tokens=["uuid", "guid", "id", "identifier"],
            positive_examples=[
                DomainExample("123e4567-e89b-12d3-a456-426614174000", "Standard UUID"),
                DomainExample("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "UUID v1"),
                DomainExample("6ba7b811-9dad-11d1-80b4-00c04fd430c8", "Another UUID"),
            ]
        ))
        
        logger.info(f"✅ Initialized domain catalog with {len(self.domains)} domains")
    
    def add_domain(self, domain: DomainDefinition, staged: bool = False):
        """Add a domain to the catalog."""
        if self.enable_embeddings:
            self._compute_embeddings(domain)
        
        target_dict = self.staged_domains if staged else self.domains
        target_dict[domain.domain_id] = domain
        
        stage_text = "staged " if staged else ""
        logger.debug(f"Added {stage_text}domain: {domain.name}")
    
    def _compute_embeddings(self, domain: DomainDefinition):
        """Compute vector embeddings for a domain."""
        if not self.embedding_model:
            return
            
        try:
            # Name embedding: combine name, aliases, and header tokens
            name_text = " ".join([domain.name] + domain.aliases + domain.header_tokens)
            domain.name_embedding = self.embedding_model.encode(name_text)
            
            # Value embedding: centroid of positive examples
            if domain.positive_examples:
                example_texts = [ex.value for ex in domain.positive_examples[:10]]  # Limit for performance
                example_embeddings = self.embedding_model.encode(example_texts)
                domain.value_embedding = np.mean(example_embeddings, axis=0)
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings for {domain.name}: {e}")
    
    def get_domain(self, domain_id: str, include_staged: bool = False) -> Optional[DomainDefinition]:
        """Get a domain by ID."""
        domain = self.domains.get(domain_id)
        if not domain and include_staged:
            domain = self.staged_domains.get(domain_id)
        return domain
    
    def search_domains(self, query: str, limit: int = 10) -> List[Tuple[DomainDefinition, float]]:
        """Search domains by text similarity."""
        if not self.enable_embeddings or not self.embedding_model:
            # Fallback to simple text matching
            return self._text_search(query, limit)
        
        try:
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            
            for domain in self.domains.values():
                if domain.name_embedding is not None:
                    similarity = np.dot(query_embedding, domain.name_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(domain.name_embedding)
                    )
                    similarities.append((domain, float(similarity)))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._text_search(query, limit)
    
    def _text_search(self, query: str, limit: int) -> List[Tuple[DomainDefinition, float]]:
        """Fallback text-based search."""
        query_lower = query.lower()
        matches = []
        
        for domain in self.domains.values():
            score = 0.0
            
            # Check name and aliases
            for name in domain.get_all_name_variants():
                if query_lower in name.lower():
                    score += 1.0
                elif any(token in name.lower() for token in query_lower.split()):
                    score += 0.5
            
            # Check header tokens
            for token in domain.header_tokens:
                if query_lower in token.lower():
                    score += 0.3
            
            if score > 0:
                matches.append((domain, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def promote_staged_domain(self, domain_id: str) -> bool:
        """Promote a staged domain to live catalog."""
        if domain_id not in self.staged_domains:
            return False
        
        domain = self.staged_domains.pop(domain_id)
        domain.provenance = "promoted"
        domain.updated_at = datetime.utcnow().isoformat()
        
        self.domains[domain_id] = domain
        logger.info(f"Promoted domain to live: {domain.name}")
        return True
    
    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return {
            "live_domains": len(self.domains),
            "staged_domains": len(self.staged_domains),
            "embeddings_enabled": self.enable_embeddings,
            "total_examples": sum(
                len(d.positive_examples) + len(d.negative_examples) 
                for d in self.domains.values()
            )
        }


# Global catalog instance
domain_catalog = DomainCatalog()


def get_domain_catalog() -> DomainCatalog:
    """Get the global domain catalog instance."""
    return domain_catalog


def initialize_domain_catalog(enable_embeddings: bool = True) -> DomainCatalog:
    """Initialize the global domain catalog."""
    global domain_catalog
    domain_catalog = DomainCatalog(enable_embeddings=enable_embeddings)
    return domain_catalog
