"""
Policy Engine for EnMapper

Phase 0: Skeleton implementation with basic policy validation framework.
Enforces mode, PII masking/redaction, RBAC, cost caps, and provider controls.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from settings import get_settings, ProcessingMode, LLMProvider


logger = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    """Policy decision outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MASK_AND_ALLOW = "mask_and_allow"


class PolicyViolation(BaseModel):
    """Policy violation details."""
    rule: str
    severity: str  # "critical", "warning", "info"
    message: str
    details: Optional[Dict[str, Any]] = None


class PolicyContext(BaseModel):
    """Context for policy evaluation."""
    user_id: str
    role: str
    mode: ProcessingMode
    provider: LLMProvider
    data_contains_pii: bool = False
    estimated_cost_usd: float = 0.0
    token_count: int = 0
    operation: str  # "ingest", "domain", "mapping", "analysis", "migration"


class PolicyResult(BaseModel):
    """Result of policy evaluation."""
    decision: PolicyDecision
    violations: List[PolicyViolation] = Field(default_factory=list)
    modifications: Dict[str, Any] = Field(default_factory=dict)
    audit_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PolicyManifest(BaseModel):
    """Policy configuration manifest (Phase 0 skeleton)."""
    version: str = "0.1.0"
    
    # PII Controls
    pii_masking_enabled: bool = True
    pii_local_only_override: bool = False
    pii_detection_threshold: float = 0.8
    
    # Cost Controls
    max_daily_cost_usd: float = 100.0
    max_tokens_per_operation: int = 50000
    cost_alert_threshold: float = 0.8
    
    # Provider Controls
    allowed_providers: List[str] = Field(default_factory=lambda: ["anthropic", "openai", "groq", "ollama"])
    denied_providers: List[str] = Field(default_factory=list)
    require_local_for_pii: bool = True
    
    # Mode Controls
    allow_data_mode: bool = True
    require_approval_for_data_mode: bool = False
    
    # RBAC
    viewer_can_see_pii: bool = False
    analyst_can_approve_operations: bool = True
    operator_can_override_local_only: bool = False


class PolicyEngine:
    """
    Policy Engine for EnMapper (Phase 0 Skeleton).
    
    Provides basic policy validation framework that will be expanded
    in later phases with full policy evaluation logic.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.manifest = PolicyManifest()
        self._load_policy_manifest()
    
    def _load_policy_manifest(self):
        """Load policy manifest from configuration."""
        # Phase 0: Use default manifest
        # Later phases will load from database or external config
        logger.info("📋 Policy Engine initialized with default manifest v0.1.0")
    
    async def evaluate_request(self, context: PolicyContext) -> PolicyResult:
        """
        Evaluate a request against current policies.
        
        Phase 0: Basic validation framework.
        Later phases will implement full policy logic.
        """
        violations = []
        modifications = {}
        decision = PolicyDecision.ALLOW
        
        # Check PII handling
        if context.data_contains_pii:
            if self.manifest.pii_masking_enabled:
                if context.provider.value in ["openai", "anthropic", "groq"]:
                    if self.manifest.require_local_for_pii and not self.manifest.pii_local_only_override:
                        violations.append(PolicyViolation(
                            rule="pii_local_only",
                            severity="critical", 
                            message="PII data requires local-only processing",
                            details={"provider": context.provider.value}
                        ))
                        decision = PolicyDecision.DENY
                    else:
                        modifications["mask_pii"] = True
                        decision = PolicyDecision.MASK_AND_ALLOW
        
        # Check cost limits
        if context.estimated_cost_usd > self.manifest.max_daily_cost_usd:
            violations.append(PolicyViolation(
                rule="cost_limit_exceeded",
                severity="critical",
                message=f"Estimated cost ${context.estimated_cost_usd:.2f} exceeds daily limit ${self.manifest.max_daily_cost_usd:.2f}",
                details={"estimated_cost": context.estimated_cost_usd, "limit": self.manifest.max_daily_cost_usd}
            ))
            decision = PolicyDecision.DENY
        
        # Check token limits
        if context.token_count > self.manifest.max_tokens_per_operation:
            violations.append(PolicyViolation(
                rule="token_limit_exceeded",
                severity="warning",
                message=f"Token count {context.token_count} exceeds operation limit {self.manifest.max_tokens_per_operation}",
                details={"token_count": context.token_count, "limit": self.manifest.max_tokens_per_operation}
            ))
        
        # Check provider allowlist
        if context.provider.value not in self.manifest.allowed_providers:
            violations.append(PolicyViolation(
                rule="provider_not_allowed",
                severity="critical",
                message=f"Provider {context.provider.value} is not in allowed list",
                details={"provider": context.provider.value, "allowed": self.manifest.allowed_providers}
            ))
            decision = PolicyDecision.DENY
        
        # Check provider denylist
        if context.provider.value in self.manifest.denied_providers:
            violations.append(PolicyViolation(
                rule="provider_denied",
                severity="critical",
                message=f"Provider {context.provider.value} is explicitly denied",
                details={"provider": context.provider.value}
            ))
            decision = PolicyDecision.DENY
        
        # Check data mode permissions
        if context.mode == ProcessingMode.DATA_MODE:
            if not self.manifest.allow_data_mode:
                violations.append(PolicyViolation(
                    rule="data_mode_disabled",
                    severity="critical",
                    message="Data mode is disabled by policy",
                    details={"mode": context.mode.value}
                ))
                decision = PolicyDecision.DENY
            elif self.manifest.require_approval_for_data_mode:
                decision = PolicyDecision.REQUIRE_APPROVAL
        
        # Build audit info
        audit_info = {
            "policy_version": self.manifest.version,
            "evaluation_time": datetime.utcnow().isoformat(),
            "context": context.dict(),
            "manifest_applied": True
        }
        
        return PolicyResult(
            decision=decision,
            violations=violations,
            modifications=modifications,
            audit_info=audit_info
        )
    
    async def check_pii_detection(self, text: str) -> Dict[str, Any]:
        """
        Basic PII detection check (Phase 0 placeholder).
        
        Returns simple pattern-based detection results.
        Later phases will implement advanced PII detection.
        """
        # Simple regex patterns for common PII
        import re
        
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        detected = {}
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = {
                    "count": len(matches),
                    "confidence": 0.9,  # High confidence for regex matches
                    "samples": matches[:3]  # First 3 matches for review
                }
        
        return {
            "contains_pii": bool(detected),
            "pii_types": detected,
            "confidence": max([d["confidence"] for d in detected.values()], default=0.0),
            "recommendation": "mask" if detected else "allow"
        }
    
    async def mask_pii(self, text: str, mask_char: str = "*") -> str:
        """
        Basic PII masking (Phase 0 implementation).
        
        Later phases will implement sophisticated masking strategies.
        """
        import re
        
        # Basic masking patterns
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', lambda m: f"{m.group()[0]}{'*' * (len(m.group()) - 2)}{m.group()[-1]}"),
            (r'\b\d{3}-\d{2}-\d{4}\b', lambda m: "***-**-****"),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', lambda m: "**** **** **** ****")
        ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, replacement, masked_text)
        
        return masked_text
    
    def get_manifest(self) -> PolicyManifest:
        """Get current policy manifest."""
        return self.manifest
    
    async def update_manifest(self, new_manifest: PolicyManifest) -> bool:
        """Update policy manifest (Phase 0: in-memory only)."""
        try:
            self.manifest = new_manifest
            logger.info(f"📋 Policy manifest updated to version {new_manifest.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to update policy manifest: {e}")
            return False
