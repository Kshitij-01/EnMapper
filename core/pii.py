"""
PII Detection and Masking System for EnMapper

This module provides comprehensive PII detection and masking capabilities
to ensure data privacy before any provider calls.
"""

import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass

import polars as pl


class PIICategory(str, Enum):
    """Categories of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    URL = "url"
    GOVERNMENT_ID = "government_id"
    FINANCIAL_ACCOUNT = "financial_account"
    MEDICAL_RECORD = "medical_record"


class MaskingStrategy(str, Enum):
    """Different strategies for masking PII."""
    REDACT = "redact"  # Replace with ***
    HASH = "hash"  # Replace with hash
    PARTIAL = "partial"  # Show partial data (e.g., first 2 chars)
    TOKEN = "token"  # Replace with a token
    FORMAT_PRESERVE = "format_preserve"  # Keep format but mask content


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    category: PIICategory
    regex: str
    confidence: float
    column_keywords: List[str]
    description: str
    
    def __post_init__(self):
        self.compiled_regex = re.compile(self.regex, re.IGNORECASE)


@dataclass
class PIIDetection:
    """Result of PII detection."""
    category: PIICategory
    confidence: float
    pattern_matched: str
    value: str
    masked_value: str
    masking_strategy: MaskingStrategy
    column_name: str


class PIIDetector:
    """Advanced PII detection system."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[PIIPattern]:
        """Initialize PII detection patterns."""
        return [
            # Email addresses
            PIIPattern(
                category=PIICategory.EMAIL,
                regex=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                confidence=0.95,
                column_keywords=['email', 'e-mail', 'mail', 'contact'],
                description="Email address pattern"
            ),
            
            # Phone numbers (various formats)
            PIIPattern(
                category=PIICategory.PHONE,
                regex=r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                confidence=0.90,
                column_keywords=['phone', 'tel', 'mobile', 'cell', 'contact', 'number'],
                description="Phone number pattern"
            ),
            
            # SSN (XXX-XX-XXXX format)
            PIIPattern(
                category=PIICategory.SSN,
                regex=r'\b(?!000|666|9\d{2})\d{3}-?(?!00)\d{2}-?(?!0000)\d{4}\b',
                confidence=0.98,
                column_keywords=['ssn', 'social', 'security', 'tax_id'],
                description="Social Security Number"
            ),
            
            # Credit card numbers
            PIIPattern(
                category=PIICategory.CREDIT_CARD,
                regex=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                confidence=0.92,
                column_keywords=['card', 'credit', 'payment', 'account'],
                description="Credit card number"
            ),
            
            # IP addresses
            PIIPattern(
                category=PIICategory.IP_ADDRESS,
                regex=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                confidence=0.85,
                column_keywords=['ip', 'address', 'host', 'server'],
                description="IP address"
            ),
            
            # URLs
            PIIPattern(
                category=PIICategory.URL,
                regex=r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                confidence=0.88,
                column_keywords=['url', 'link', 'website', 'domain'],
                description="URL pattern"
            ),
            
            # Date of birth patterns
            PIIPattern(
                category=PIICategory.DATE_OF_BIRTH,
                regex=r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
                confidence=0.80,
                column_keywords=['birth', 'dob', 'birthday', 'born'],
                description="Date of birth"
            ),
            
            # Driver's License patterns (US format)
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b[A-Z]{1,2}[0-9]{6,8}\b|\b[0-9]{8,10}\b',
                confidence=0.75,
                column_keywords=['license', 'dl', 'driver', 'id_number', 'state_id'],
                description="Driver's license or state ID"
            ),
            
            # Passport numbers
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b[A-Z]{1,2}[0-9]{6,9}\b',
                confidence=0.85,
                column_keywords=['passport', 'passport_number', 'travel_doc'],
                description="Passport number"
            ),
            
            # Tax ID/EIN patterns
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b\d{2}-\d{7}\b',
                confidence=0.90,
                column_keywords=['ein', 'tax_id', 'employer_id', 'federal_id'],
                description="Employer Identification Number (EIN)"
            ),
            
            # Bank account numbers
            PIIPattern(
                category=PIICategory.FINANCIAL_ACCOUNT,
                regex=r'\b\d{8,17}\b',
                confidence=0.70,
                column_keywords=['account', 'bank', 'routing', 'aba', 'swift'],
                description="Bank account number"
            ),
            
            # IBAN patterns
            PIIPattern(
                category=PIICategory.FINANCIAL_ACCOUNT,
                regex=r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
                confidence=0.95,
                column_keywords=['iban', 'international', 'bank_account'],
                description="International Bank Account Number (IBAN)"
            ),
            
            # Medical Record Numbers
            PIIPattern(
                category=PIICategory.MEDICAL_RECORD,
                regex=r'\b(?:MRN|MR|HOSP)[:\-]?\s*[A-Z0-9]{6,12}\b',
                confidence=0.85,
                column_keywords=['mrn', 'medical', 'patient', 'hospital', 'record'],
                description="Medical Record Number"
            ),
            
            # Health Insurance Numbers
            PIIPattern(
                category=PIICategory.MEDICAL_RECORD,
                regex=r'\b[A-Z]{3}\d{9}\b|\b\d{3}[A-Z]{2}\d{4}\b',
                confidence=0.80,
                column_keywords=['insurance', 'policy', 'member_id', 'health_id'],
                description="Health insurance number"
            ),
            
            # Medicare/Medicaid numbers
            PIIPattern(
                category=PIICategory.MEDICAL_RECORD,
                regex=r'\b\d{4}[A-Z]{2}\d{4}[A-Z]?\b|\b[A-Z]\d{8}[A-Z]?\b',
                confidence=0.88,
                column_keywords=['medicare', 'medicaid', 'cms', 'health_insurance'],
                description="Medicare/Medicaid number"
            ),
            
            # Student ID numbers
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b(?:STU|STUD|ID)[:\-]?\s*[A-Z0-9]{6,12}\b',
                confidence=0.75,
                column_keywords=['student', 'school', 'university', 'education', 'student_id'],
                description="Student identification number"
            ),
            
            # Vehicle Identification Numbers (VIN)
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b[A-HJ-NPR-Z0-9]{17}\b',
                confidence=0.90,
                column_keywords=['vin', 'vehicle', 'car', 'auto', 'registration'],
                description="Vehicle Identification Number"
            ),
            
            # ITIN (Individual Taxpayer Identification Number)
            PIIPattern(
                category=PIICategory.GOVERNMENT_ID,
                regex=r'\b9\d{2}-[789]\d-\d{4}\b',
                confidence=0.95,
                column_keywords=['itin', 'tax', 'taxpayer', 'individual_id'],
                description="Individual Taxpayer Identification Number"
            ),
        ]
    
    def detect_pii_in_value(self, value: Any, column_name: str = "") -> List[PIIDetection]:
        """Detect PII in a single value."""
        if value is None:
            return []
        
        value_str = str(value)
        detections = []
        
        for pattern in self.patterns:
            matches = pattern.compiled_regex.findall(value_str)
            if matches:
                # Boost confidence if column name matches keywords
                confidence = pattern.confidence
                if any(keyword in column_name.lower() for keyword in pattern.column_keywords):
                    confidence = min(0.99, confidence + 0.05)
                
                for match in matches:
                    match_str = match if isinstance(match, str) else ''.join(match)
                    masked_value = self._mask_value(match_str, pattern.category)
                    
                    detection = PIIDetection(
                        category=pattern.category,
                        confidence=confidence,
                        pattern_matched=pattern.regex,
                        value=match_str,
                        masked_value=masked_value,
                        masking_strategy=self._get_masking_strategy(pattern.category),
                        column_name=column_name
                    )
                    detections.append(detection)
        
        return detections
    
    def detect_pii_in_column(self, df: pl.DataFrame, column_name: str) -> Dict[str, List[PIIDetection]]:
        """Detect PII in an entire column."""
        column_data = df[column_name]
        pii_detections = {}
        
        for i, value in enumerate(column_data):
            detections = self.detect_pii_in_value(value, column_name)
            if detections:
                pii_detections[f"row_{i}"] = detections
        
        return pii_detections
    
    def scan_dataframe(self, df: pl.DataFrame) -> Dict[str, Dict[str, List[PIIDetection]]]:
        """Scan entire dataframe for PII."""
        results = {}
        
        for column_name in df.columns:
            column_detections = self.detect_pii_in_column(df, column_name)
            if column_detections:
                results[column_name] = column_detections
        
        return results
    
    def _get_masking_strategy(self, category: PIICategory) -> MaskingStrategy:
        """Get appropriate masking strategy for PII category."""
        strategy_map = {
            PIICategory.EMAIL: MaskingStrategy.PARTIAL,
            PIICategory.PHONE: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.SSN: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.CREDIT_CARD: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.NAME: MaskingStrategy.REDACT,
            PIICategory.ADDRESS: MaskingStrategy.REDACT,
            PIICategory.DATE_OF_BIRTH: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.IP_ADDRESS: MaskingStrategy.PARTIAL,
            PIICategory.URL: MaskingStrategy.PARTIAL,
            PIICategory.GOVERNMENT_ID: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.FINANCIAL_ACCOUNT: MaskingStrategy.FORMAT_PRESERVE,
            PIICategory.MEDICAL_RECORD: MaskingStrategy.HASH,
        }
        return strategy_map.get(category, MaskingStrategy.REDACT)
    
    def _mask_value(self, value: str, category: PIICategory) -> str:
        """Apply masking to a PII value."""
        strategy = self._get_masking_strategy(category)
        
        if strategy == MaskingStrategy.REDACT:
            return "***REDACTED***"
        
        elif strategy == MaskingStrategy.HASH:
            return f"HASH_{hashlib.sha256(value.encode()).hexdigest()[:8]}"
        
        elif strategy == MaskingStrategy.PARTIAL:
            if category == PIICategory.EMAIL:
                if '@' in value:
                    local, domain = value.split('@', 1)
                    masked_local = local[:2] + '*' * (len(local) - 2) if len(local) > 2 else '***'
                    domain_parts = domain.split('.')
                    if len(domain_parts) >= 2:
                        masked_domain = domain_parts[0][:1] + '*' * (len(domain_parts[0]) - 1) + '.' + domain_parts[-1]
                    else:
                        masked_domain = '***'
                    return f"{masked_local}@{masked_domain}"
                return "***@***"
            
            elif category == PIICategory.IP_ADDRESS:
                parts = value.split('.')
                if len(parts) == 4:
                    return f"{parts[0]}.{parts[1]}.***.**"
                return "***.***.***.**"
            
            elif category == PIICategory.URL:
                return value[:10] + "***" if len(value) > 10 else "***"
            
            else:
                return value[:2] + '*' * (len(value) - 2) if len(value) > 2 else '***'
        
        elif strategy == MaskingStrategy.FORMAT_PRESERVE:
            if category == PIICategory.PHONE:
                # Preserve phone format but mask digits
                masked = re.sub(r'\d', '*', value)
                return masked
            
            elif category == PIICategory.SSN:
                # XXX-XX-XXXX -> ***-**-****
                return re.sub(r'\d', '*', value)
            
            elif category == PIICategory.CREDIT_CARD:
                # Show last 4 digits
                digits_only = re.sub(r'\D', '', value)
                if len(digits_only) >= 4:
                    masked_digits = '*' * (len(digits_only) - 4) + digits_only[-4:]
                    return re.sub(r'\d+', masked_digits, value)
                return re.sub(r'\d', '*', value)
            
            elif category == PIICategory.DATE_OF_BIRTH:
                # MM/DD/YYYY -> **/**/****
                return re.sub(r'\d', '*', value)
            
            else:
                return re.sub(r'\w', '*', value)
        
        elif strategy == MaskingStrategy.TOKEN:
            return f"PII_TOKEN_{category.value.upper()}_{hash(value) % 10000:04d}"
        
        return "***"


class PIIMasker:
    """High-level PII masking orchestrator."""
    
    def __init__(self):
        self.detector = PIIDetector()
    
    def mask_dataframe(self, df: pl.DataFrame, aggressive: bool = False) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Mask PII in a dataframe and return the masked dataframe plus metadata.
        
        Args:
            df: Input dataframe
            aggressive: If True, use more aggressive detection (higher false positives)
            
        Returns:
            Tuple of (masked_dataframe, masking_metadata)
        """
        # Scan for PII
        pii_scan_results = self.detector.scan_dataframe(df)
        
        # Build masking metadata
        masking_metadata = {
            "pii_fields_detected": list(pii_scan_results.keys()),
            "total_pii_instances": sum(
                len(row_detections) 
                for col_detections in pii_scan_results.values() 
                for row_detections in col_detections.values()
            ),
            "pii_categories_found": list(set(
                detection.category 
                for col_detections in pii_scan_results.values()
                for row_detections in col_detections.values()
                for detection in row_detections
            )),
            "masking_applied": {},
            "detection_confidence": self._calculate_overall_confidence(pii_scan_results)
        }
        
        # Apply masking
        masked_df = df.clone()
        
        for column_name, column_detections in pii_scan_results.items():
            masking_metadata["masking_applied"][column_name] = []
            
            # Collect all unique PII values in this column for masking
            values_to_mask = set()
            mask_mappings = {}
            
            for row_detections in column_detections.values():
                for detection in row_detections:
                    values_to_mask.add(detection.value)
                    mask_mappings[detection.value] = detection.masked_value
                    masking_metadata["masking_applied"][column_name].append({
                        "category": detection.category,
                        "confidence": detection.confidence,
                        "strategy": detection.masking_strategy
                    })
            
            # Apply masking using Polars string operations
            if values_to_mask:
                masked_column = masked_df[column_name].cast(pl.Utf8)
                
                for original_value, masked_value in mask_mappings.items():
                    masked_column = masked_column.str.replace_all(
                        re.escape(str(original_value)), 
                        masked_value, 
                        literal=True
                    )
                
                masked_df = masked_df.with_columns(masked_column.alias(column_name))
        
        return masked_df, masking_metadata
    
    def _calculate_overall_confidence(self, pii_scan_results: Dict) -> float:
        """Calculate overall confidence score for PII detection."""
        if not pii_scan_results:
            return 0.0
        
        total_detections = 0
        total_confidence = 0.0
        
        for col_detections in pii_scan_results.values():
            for row_detections in col_detections.values():
                for detection in row_detections:
                    total_detections += 1
                    total_confidence += detection.confidence
        
        return total_confidence / total_detections if total_detections > 0 else 0.0
    
    def validate_masking_policy(self, df: pl.DataFrame, policy_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that masking meets policy requirements."""
        # Scan the supposedly masked dataframe for any remaining PII
        remaining_pii = self.detector.scan_dataframe(df)
        
        validation_result = {
            "compliant": len(remaining_pii) == 0,
            "remaining_pii_fields": list(remaining_pii.keys()),
            "remaining_pii_count": sum(
                len(row_detections) 
                for col_detections in remaining_pii.values() 
                for row_detections in col_detections.values()
            ),
            "policy_requirements_met": True,  # TODO: implement specific policy checks
            "recommendations": []
        }
        
        if not validation_result["compliant"]:
            validation_result["recommendations"].append(
                "Additional masking required for complete PII removal"
            )
        
        return validation_result


# Global PII masker instance
pii_masker = PIIMasker()
