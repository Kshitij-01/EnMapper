"""
Artifact Generation and Storage System for EnMapper

This module handles the creation, storage, and retrieval of various artifacts
produced during data processing runs, including Catalog v1 and Sample Pack v1.
"""

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field
import polars as pl


class ArtifactType(str, Enum):
    """Types of artifacts that can be generated."""
    CATALOG_V1 = "catalog_v1"
    SAMPLE_PACK_V1 = "sample_pack_v1"
    DOMAIN_ASSIGNMENTS = "domain_assignments_v1"
    TRANSFORM_DSL = "transform_dsl_v1"
    MAPPING_RESULTS = "mapping_results_v1"


class ArtifactStatus(str, Enum):
    """Status of artifact generation/storage."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ColumnInfo(BaseModel):
    """Information about a single column in a dataset."""
    name: str
    original_name: str
    data_type: str
    nullable: bool
    unique_values: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[Union[str, int, float]] = None
    max_value: Optional[Union[str, int, float]] = None
    sample_values: List[Any] = Field(default_factory=list)
    
    # Standardization info
    standardized_name: Optional[str] = None
    standardization_applied: List[str] = Field(default_factory=list)
    
    # Inferred metadata
    detected_encoding: Optional[str] = None
    inferred_format: Optional[str] = None
    confidence_score: Optional[float] = None


class DataSourceInfo(BaseModel):
    """Information about the source of the data."""
    source_type: str  # 'file' or 'sql'
    source_identifier: str  # file path or connection string
    dialect: Optional[str] = None
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    has_header: Optional[bool] = None
    total_rows: Optional[int] = None
    total_size_bytes: Optional[int] = None


class CatalogV1(BaseModel):
    """Catalog v1 artifact containing schema and metadata."""
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Data source information
    data_source: DataSourceInfo
    
    # Schema information
    columns: List[ColumnInfo]
    total_columns: int
    
    # Processing metadata
    standardization_rules_applied: List[str] = Field(default_factory=list)
    dialect_detection_confidence: Optional[float] = None
    schema_inference_method: str = "polars_auto"
    
    # Quality metrics
    data_quality_score: Optional[float] = None
    completeness_ratio: Optional[float] = None
    uniqueness_ratio: Optional[float] = None
    
    def get_column_by_name(self, name: str) -> Optional[ColumnInfo]:
        """Get column information by name."""
        for col in self.columns:
            if col.name == name or col.original_name == name:
                return col
        return None
    
    def get_columns_by_type(self, data_type: str) -> List[ColumnInfo]:
        """Get all columns of a specific data type."""
        return [col for col in self.columns if col.data_type == data_type]


class MaskedSample(BaseModel):
    """A single masked sample row."""
    original_index: int
    masked_values: Dict[str, Any]
    pii_fields_masked: List[str] = Field(default_factory=list)
    masking_applied: Dict[str, str] = Field(default_factory=dict)  # field -> mask_type


class SamplePackV1(BaseModel):
    """Sample Pack v1 artifact containing masked sample data."""
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Reference to catalog
    catalog_artifact_id: str
    
    # Sampling configuration
    sampling_method: str = "stratified"
    sample_size: int
    sampling_seed: int
    total_population: int
    
    # Masked samples
    samples: List[MaskedSample]
    
    # PII masking info
    pii_detection_confidence: float = 0.0
    pii_fields_detected: List[str] = Field(default_factory=list)
    masking_rules_applied: List[str] = Field(default_factory=list)
    
    # Quality metrics
    representativeness_score: Optional[float] = None
    privacy_safety_score: Optional[float] = None


class Artifact(BaseModel):
    """Generic artifact metadata."""
    artifact_id: str
    run_id: str
    artifact_type: ArtifactType
    status: ArtifactStatus
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Storage info
    storage_path: Optional[str] = None
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    
    # Processing info
    generation_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactManager:
    """Manages artifact generation, storage, and retrieval."""
    
    def __init__(self, storage_base_path: str = "artifacts"):
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(exist_ok=True, parents=True)
        
        # In-memory registry for this session
        self.artifacts: Dict[str, Artifact] = {}
    
    def _generate_storage_path(self, run_id: str, artifact_type: ArtifactType, artifact_id: str) -> Path:
        """Generate storage path for an artifact."""
        return self.storage_base_path / run_id / f"{artifact_type.value}_{artifact_id}.json"
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def store_artifact(self, artifact_data: Union[CatalogV1, SamplePackV1], artifact_type: ArtifactType) -> Artifact:
        """Store an artifact and return metadata."""
        start_time = datetime.utcnow()
        
        # Create artifact metadata
        artifact = Artifact(
            artifact_id=artifact_data.artifact_id,
            run_id=artifact_data.run_id,
            artifact_type=artifact_type,
            status=ArtifactStatus.GENERATING
        )
        
        try:
            # Serialize the artifact
            content = artifact_data.model_dump_json(indent=2)
            
            # Calculate hash and size
            content_hash = self._calculate_content_hash(content)
            size_bytes = len(content.encode('utf-8'))
            
            # Determine storage path
            storage_path = self._generate_storage_path(
                artifact_data.run_id, 
                artifact_type, 
                artifact_data.artifact_id
            )
            
            # Ensure directory exists
            storage_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Write to disk
            with open(storage_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update artifact metadata
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            artifact.status = ArtifactStatus.COMPLETED
            artifact.storage_path = str(storage_path)
            artifact.content_hash = content_hash
            artifact.size_bytes = size_bytes
            artifact.generation_time_seconds = generation_time
            artifact.updated_at = datetime.utcnow().isoformat()
            
            # Store in registry
            self.artifacts[artifact.artifact_id] = artifact
            
            return artifact
            
        except Exception as e:
            artifact.status = ArtifactStatus.FAILED
            artifact.error_message = str(e)
            artifact.updated_at = datetime.utcnow().isoformat()
            
            # Store failed artifact in registry too
            self.artifacts[artifact.artifact_id] = artifact
            raise
    
    def get_artifact_metadata(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact metadata by ID."""
        return self.artifacts.get(artifact_id)
    
    def load_artifact(self, artifact_id: str, artifact_type: ArtifactType) -> Optional[Union[CatalogV1, SamplePackV1]]:
        """Load an artifact from storage."""
        artifact = self.get_artifact_metadata(artifact_id)
        if not artifact or not artifact.storage_path:
            return None
        
        try:
            with open(artifact.storage_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse based on type
            if artifact_type == ArtifactType.CATALOG_V1:
                return CatalogV1.model_validate_json(content)
            elif artifact_type == ArtifactType.SAMPLE_PACK_V1:
                return SamplePackV1.model_validate_json(content)
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
                
        except Exception as e:
            print(f"Error loading artifact {artifact_id}: {e}")
            return None
    
    def list_artifacts_for_run(self, run_id: str) -> List[Artifact]:
        """List all artifacts for a specific run."""
        return [artifact for artifact in self.artifacts.values() if artifact.run_id == run_id]
    
    def list_artifacts_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """List all artifacts of a specific type."""
        return [artifact for artifact in self.artifacts.values() if artifact.artifact_type == artifact_type]


class CatalogGenerator:
    """Generates Catalog v1 artifacts from processed data."""
    
    @staticmethod
    def generate_from_dataframe(
        df: pl.DataFrame, 
        run_id: str, 
        data_source: DataSourceInfo,
        standardization_rules: List[str] = None
    ) -> CatalogV1:
        """Generate a Catalog v1 artifact from a Polars DataFrame."""
        
        columns = []
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Basic column info
            col_info = ColumnInfo(
                name=col_name,
                original_name=col_name,  # TODO: track original names through standardization
                data_type=str(col_data.dtype),
                nullable=col_data.null_count() > 0,
                null_count=col_data.null_count(),
            )
            
            # Add statistics for non-null values
            non_null_values = col_data.drop_nulls()
            if len(non_null_values) > 0:
                col_info.unique_values = non_null_values.n_unique()
                
                # Type-specific statistics
                if col_data.dtype.is_numeric():
                    try:
                        col_info.min_value = float(non_null_values.min())
                        col_info.max_value = float(non_null_values.max())
                    except:
                        pass
                
                # Sample values (first 5 unique non-null values)
                try:
                    sample_values = non_null_values.unique().head(5).to_list()
                    col_info.sample_values = sample_values
                except:
                    pass
            
            columns.append(col_info)
        
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        completeness_ratio = (total_cells - null_cells) / total_cells if total_cells > 0 else 0.0
        
        # Create catalog
        catalog = CatalogV1(
            run_id=run_id,
            data_source=data_source,
            columns=columns,
            total_columns=len(columns),
            standardization_rules_applied=standardization_rules or [],
            schema_inference_method="polars_auto",
            completeness_ratio=completeness_ratio
        )
        
        return catalog


class SamplePackGenerator:
    """Generates Sample Pack v1 artifacts with PII masking."""
    
    @staticmethod
    def generate_stratified_sample(
        df: pl.DataFrame,
        run_id: str,
        catalog_artifact_id: str,
        sample_size: int = 100,
        seed: int = 42
    ) -> SamplePackV1:
        """Generate a stratified sample with basic masking."""
        
        # For now, use simple random sampling
        # TODO: Implement proper stratified sampling based on column types/values
        
        total_rows = len(df)
        actual_sample_size = min(sample_size, total_rows)
        
        # Sample rows
        if total_rows > actual_sample_size:
            sampled_df = df.sample(n=actual_sample_size, seed=seed)
        else:
            sampled_df = df
        
        # Generate masked samples
        samples = []
        pii_fields_detected = []
        
        for i, row in enumerate(sampled_df.iter_rows(named=True)):
            masked_values = {}
            pii_fields_masked = []
            masking_applied = {}
            
            for col_name, value in row.items():
                # Basic PII detection and masking
                masked_value, is_pii, mask_type = SamplePackGenerator._mask_if_pii(col_name, value)
                
                masked_values[col_name] = masked_value
                
                if is_pii:
                    pii_fields_masked.append(col_name)
                    masking_applied[col_name] = mask_type
                    if col_name not in pii_fields_detected:
                        pii_fields_detected.append(col_name)
            
            sample = MaskedSample(
                original_index=i,  # This would be the actual row index in stratified sampling
                masked_values=masked_values,
                pii_fields_masked=pii_fields_masked,
                masking_applied=masking_applied
            )
            samples.append(sample)
        
        # Create sample pack
        sample_pack = SamplePackV1(
            run_id=run_id,
            catalog_artifact_id=catalog_artifact_id,
            sampling_method="stratified",
            sample_size=actual_sample_size,
            sampling_seed=seed,
            total_population=total_rows,
            samples=samples,
            pii_fields_detected=pii_fields_detected,
            pii_detection_confidence=0.8,  # Basic confidence score
            masking_rules_applied=["email_mask", "phone_mask", "ssn_mask"]
        )
        
        return sample_pack
    
    @staticmethod
    def _mask_if_pii(column_name: str, value: Any) -> tuple[Any, bool, Optional[str]]:
        """Basic PII detection and masking."""
        if value is None:
            return value, False, None
        
        value_str = str(value).lower()
        column_lower = column_name.lower()
        
        # Email detection
        if '@' in value_str and '.' in value_str:
            return "***@***.***", True, "email_mask"
        
        # Phone number detection
        if any(keyword in column_lower for keyword in ['phone', 'mobile', 'tel']):
            return "***-***-****", True, "phone_mask"
        
        # SSN detection
        if any(keyword in column_lower for keyword in ['ssn', 'social']):
            return "***-**-****", True, "ssn_mask"
        
        # Credit card detection (basic)
        if len(value_str.replace('-', '').replace(' ', '')) == 16 and value_str.replace('-', '').replace(' ', '').isdigit():
            return "****-****-****-****", True, "credit_card_mask"
        
        # Name detection
        if any(keyword in column_lower for keyword in ['name', 'first', 'last', 'fname', 'lname']):
            return "***", True, "name_mask"
        
        return value, False, None


# Global artifact manager instance
artifact_manager = ArtifactManager()


