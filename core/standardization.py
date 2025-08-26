"""
Standardization Shim for Phase 1

Applies minimal Transform DSL operations to canonicalize headers, case, whitespace, and data types.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import polars as pl
import pandas as pd

from .inference import TableSchema, ColumnInfo, DataType

logger = logging.getLogger(__name__)


class TransformOperation(str, Enum):
    """Supported Transform DSL operations (whitelisted)."""
    TRIM = "trim"
    CASE_LOWER = "case_lower"
    CASE_UPPER = "case_upper"
    CASE_TITLE = "case_title"
    COLLAPSE_WHITESPACE = "collapse_whitespace"
    REGEX_EXTRACT = "regex_extract"
    REGEX_SPLIT = "regex_split"
    CONCAT = "concat"
    MAP_DICT = "map_dict"
    UNIT_CONVERT = "unit_convert"
    DATE_PARSE = "date_parse"
    DATE_FORMAT = "date_format"
    TYPE_CAST = "type_cast"
    SAFE_MATH = "safe_math"
    STANDARDIZE_NAME = "standardize_name"


@dataclass
class TransformStep:
    """A single transformation step."""
    operation: TransformOperation
    column: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class TransformPlan:
    """A plan containing multiple transformation steps."""
    steps: List[TransformStep]
    target_schema: Optional[TableSchema] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StandardizationResult:
    """Result of standardization process."""
    success: bool
    transformed_schema: Optional[TableSchema] = None
    transform_plan: Optional[TransformPlan] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class StandardizationShim:
    """
    Standardization Shim that applies minimal Transform DSL operations.
    
    Responsibilities:
    - Canonicalize column headers (case, whitespace, special characters)
    - Standardize data types
    - Apply basic data cleaning operations
    - Generate Transform DSL plans
    """
    
    def __init__(self):
        self.column_name_patterns = {
            # Common patterns for standardization
            'id': r'.*\b(id|identifier|key)\b.*',
            'name': r'.*\b(name|title|label)\b.*',
            'email': r'.*\b(email|e_mail|mail)\b.*',
            'phone': r'.*\b(phone|tel|telephone|mobile)\b.*',
            'date': r'.*\b(date|created|updated|modified)\b.*',
            'amount': r'.*\b(amount|price|cost|value|total)\b.*',
            'address': r'.*\b(address|street|city|state|zip|postal)\b.*'
        }
    
    def auto_standardize(self, schema: TableSchema, sample_data: Optional[List[Dict[str, Any]]] = None) -> StandardizationResult:
        """
        Automatically generate and apply standardization transformations.
        
        Args:
            schema: Input table schema
            sample_data: Optional sample data for analysis
        
        Returns:
            StandardizationResult with transformed schema and plan
        """
        try:
            logger.info(f"Starting auto-standardization for table: {schema.name}")
            
            # Generate transformation plan
            transform_plan = self._generate_auto_plan(schema, sample_data)
            
            # Apply transformations to schema
            transformed_schema = self._apply_plan_to_schema(schema, transform_plan)
            
            # Apply transformations to sample data if provided
            transformed_sample = None
            if sample_data:
                transformed_sample = self._apply_plan_to_data(sample_data, transform_plan)
            
            logger.info(f"Standardization completed: {len(transform_plan.steps)} transformations applied")
            
            return StandardizationResult(
                success=True,
                transformed_schema=transformed_schema,
                transform_plan=transform_plan,
                sample_data=transformed_sample
            )
            
        except Exception as e:
            logger.error(f"Standardization failed: {e}")
            return StandardizationResult(
                success=False,
                error_message=str(e)
            )
    
    def _generate_auto_plan(self, schema: TableSchema, sample_data: Optional[List[Dict[str, Any]]]) -> TransformPlan:
        """Generate automatic transformation plan."""
        steps = []
        
        for column in schema.columns:
            # 1. Standardize column names
            if column.name != column.original_name or self._needs_name_standardization(column.original_name):
                steps.append(TransformStep(
                    operation=TransformOperation.STANDARDIZE_NAME,
                    column=column.original_name,
                    parameters={"target_name": self._standardize_column_name(column.original_name)}
                ))
            
            # 2. Handle string columns
            if column.data_type == DataType.STRING:
                # Trim whitespace
                steps.append(TransformStep(
                    operation=TransformOperation.TRIM,
                    column=column.original_name
                ))
                
                # Collapse multiple whitespaces
                steps.append(TransformStep(
                    operation=TransformOperation.COLLAPSE_WHITESPACE,
                    column=column.original_name
                ))
                
                # Determine case normalization based on content
                case_operation = self._determine_case_operation(column, sample_data)
                if case_operation:
                    steps.append(TransformStep(
                        operation=case_operation,
                        column=column.original_name
                    ))
            
            # 3. Handle date/datetime columns
            if column.data_type in [DataType.DATE, DataType.DATETIME]:
                # Ensure consistent date format
                steps.append(TransformStep(
                    operation=TransformOperation.DATE_FORMAT,
                    column=column.original_name,
                    parameters={"format": "ISO8601"}
                ))
            
            # 4. Handle type casting if needed
            target_type = self._determine_target_type(column, sample_data)
            if target_type and target_type != column.data_type:
                steps.append(TransformStep(
                    operation=TransformOperation.TYPE_CAST,
                    column=column.original_name,
                    parameters={"target_type": target_type.value}
                ))
        
        return TransformPlan(
            steps=steps,
            target_schema=schema,
            metadata={
                "auto_generated": True,
                "total_operations": len(steps),
                "operation_types": list(set(step.operation.value for step in steps))
            }
        )
    
    def _apply_plan_to_schema(self, schema: TableSchema, plan: TransformPlan) -> TableSchema:
        """Apply transformation plan to schema."""
        # Create a copy of the schema with transformed columns
        transformed_columns = []
        
        for column in schema.columns:
            # Apply relevant transformations to this column
            transformed_column = ColumnInfo(
                name=column.name,
                original_name=column.original_name,
                data_type=column.data_type,
                nullable=column.nullable,
                unique_count=column.unique_count,
                null_count=column.null_count,
                sample_values=column.sample_values,
                min_value=column.min_value,
                max_value=column.max_value,
                avg_length=column.avg_length,
                regex_patterns=column.regex_patterns
            )
            
            # Apply column-specific transformations
            for step in plan.steps:
                if step.column == column.original_name:
                    transformed_column = self._apply_step_to_column(transformed_column, step)
            
            transformed_columns.append(transformed_column)
        
        # Create transformed schema
        transformed_schema = TableSchema(
            name=schema.name,
            columns=transformed_columns,
            row_count=schema.row_count,
            file_size_bytes=schema.file_size_bytes,
            encoding=schema.encoding,
            dialect=schema.dialect,
            inferred_format=schema.inferred_format,
            confidence=schema.confidence
        )
        
        return transformed_schema
    
    def _apply_plan_to_data(self, sample_data: List[Dict[str, Any]], plan: TransformPlan) -> List[Dict[str, Any]]:
        """Apply transformation plan to sample data."""
        if not sample_data:
            return []
        
        # Convert to DataFrame for easier manipulation
        try:
            df = pl.DataFrame(sample_data)
        except Exception:
            # Fallback to pandas
            df = pd.DataFrame(sample_data)
            return self._apply_plan_pandas(df, plan)
        
        # Apply transformations using Polars
        for step in plan.steps:
            df = self._apply_step_to_dataframe(df, step)
        
        # Convert back to list of dicts
        return df.to_dicts()
    
    def _apply_plan_pandas(self, df: pd.DataFrame, plan: TransformPlan) -> List[Dict[str, Any]]:
        """Apply transformation plan using pandas (fallback)."""
        for step in plan.steps:
            if step.column in df.columns:
                df = self._apply_step_to_pandas_df(df, step)
        
        return df.to_dict('records')
    
    def _apply_step_to_column(self, column: ColumnInfo, step: TransformStep) -> ColumnInfo:
        """Apply a transformation step to a column definition."""
        if step.operation == TransformOperation.STANDARDIZE_NAME:
            column.name = step.parameters.get("target_name", column.name)
        elif step.operation == TransformOperation.TYPE_CAST:
            target_type_str = step.parameters.get("target_type")
            if target_type_str:
                try:
                    column.data_type = DataType(target_type_str)
                except ValueError:
                    logger.warning(f"Unknown target type: {target_type_str}")
        
        return column
    
    def _apply_step_to_dataframe(self, df: pl.DataFrame, step: TransformStep) -> pl.DataFrame:
        """Apply a transformation step to a Polars DataFrame."""
        if step.column not in df.columns:
            return df
        
        try:
            if step.operation == TransformOperation.TRIM:
                df = df.with_columns(pl.col(step.column).str.strip())
            
            elif step.operation == TransformOperation.CASE_LOWER:
                df = df.with_columns(pl.col(step.column).str.to_lowercase())
            
            elif step.operation == TransformOperation.CASE_UPPER:
                df = df.with_columns(pl.col(step.column).str.to_uppercase())
            
            elif step.operation == TransformOperation.CASE_TITLE:
                df = df.with_columns(pl.col(step.column).str.to_titlecase())
            
            elif step.operation == TransformOperation.COLLAPSE_WHITESPACE:
                df = df.with_columns(
                    pl.col(step.column).str.replace_all(r'\s+', ' ')
                )
            
            elif step.operation == TransformOperation.STANDARDIZE_NAME:
                # Rename column
                new_name = step.parameters.get("target_name", step.column)
                if new_name != step.column:
                    df = df.rename({step.column: new_name})
            
            elif step.operation == TransformOperation.TYPE_CAST:
                target_type = step.parameters.get("target_type")
                if target_type == "integer":
                    df = df.with_columns(pl.col(step.column).cast(pl.Int64, strict=False))
                elif target_type == "float":
                    df = df.with_columns(pl.col(step.column).cast(pl.Float64, strict=False))
                elif target_type == "string":
                    df = df.with_columns(pl.col(step.column).cast(pl.Utf8))
                elif target_type == "boolean":
                    df = df.with_columns(pl.col(step.column).cast(pl.Boolean, strict=False))
            
        except Exception as e:
            logger.warning(f"Failed to apply {step.operation} to {step.column}: {e}")
        
        return df
    
    def _apply_step_to_pandas_df(self, df: pd.DataFrame, step: TransformStep) -> pd.DataFrame:
        """Apply a transformation step to a pandas DataFrame (fallback)."""
        try:
            if step.operation == TransformOperation.TRIM:
                if df[step.column].dtype == 'object':
                    df[step.column] = df[step.column].astype(str).str.strip()
            
            elif step.operation == TransformOperation.CASE_LOWER:
                if df[step.column].dtype == 'object':
                    df[step.column] = df[step.column].astype(str).str.lower()
            
            elif step.operation == TransformOperation.CASE_UPPER:
                if df[step.column].dtype == 'object':
                    df[step.column] = df[step.column].astype(str).str.upper()
            
            elif step.operation == TransformOperation.CASE_TITLE:
                if df[step.column].dtype == 'object':
                    df[step.column] = df[step.column].astype(str).str.title()
            
            elif step.operation == TransformOperation.COLLAPSE_WHITESPACE:
                if df[step.column].dtype == 'object':
                    df[step.column] = df[step.column].astype(str).str.replace(r'\s+', ' ', regex=True)
            
            elif step.operation == TransformOperation.STANDARDIZE_NAME:
                new_name = step.parameters.get("target_name", step.column)
                if new_name != step.column:
                    df = df.rename(columns={step.column: new_name})
            
            elif step.operation == TransformOperation.TYPE_CAST:
                target_type = step.parameters.get("target_type")
                if target_type == "integer":
                    df[step.column] = pd.to_numeric(df[step.column], errors='coerce').astype('Int64')
                elif target_type == "float":
                    df[step.column] = pd.to_numeric(df[step.column], errors='coerce')
                elif target_type == "string":
                    df[step.column] = df[step.column].astype(str)
                elif target_type == "boolean":
                    df[step.column] = df[step.column].astype('boolean')
        
        except Exception as e:
            logger.warning(f"Failed to apply {step.operation} to {step.column}: {e}")
        
        return df
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize a column name."""
        # Convert to lowercase
        standardized = name.lower()
        
        # Replace special characters with underscores
        standardized = re.sub(r'[^\w\s]', '_', standardized)
        
        # Collapse multiple underscores and whitespace
        standardized = re.sub(r'[\s_]+', '_', standardized)
        
        # Remove leading/trailing underscores
        standardized = standardized.strip('_')
        
        # Ensure it's not empty
        if not standardized:
            standardized = "unnamed_column"
        
        return standardized
    
    def _needs_name_standardization(self, name: str) -> bool:
        """Check if a column name needs standardization."""
        # Check for special characters, spaces, or mixed case
        return bool(re.search(r'[^\w]|[A-Z]', name)) or ' ' in name
    
    def _determine_case_operation(self, column: ColumnInfo, sample_data: Optional[List[Dict[str, Any]]]) -> Optional[TransformOperation]:
        """Determine appropriate case operation based on column content."""
        if not sample_data or column.data_type != DataType.STRING:
            return None
        
        # Analyze sample values to determine common case pattern
        values = []
        for row in sample_data[:100]:  # Sample first 100 rows
            value = row.get(column.original_name)
            if isinstance(value, str) and value.strip():
                values.append(value)
        
        if not values:
            return None
        
        # Count different case patterns
        upper_count = sum(1 for v in values if v.isupper())
        lower_count = sum(1 for v in values if v.islower())
        title_count = sum(1 for v in values if v.istitle())
        
        # Determine majority pattern
        total = len(values)
        if lower_count / total > 0.7:
            return None  # Already mostly lowercase
        elif upper_count / total > 0.7:
            return TransformOperation.CASE_LOWER
        elif title_count / total > 0.7:
            return TransformOperation.CASE_LOWER
        else:
            # Mixed case - default to lowercase for consistency
            return TransformOperation.CASE_LOWER
    
    def _determine_target_type(self, column: ColumnInfo, sample_data: Optional[List[Dict[str, Any]]]) -> Optional[DataType]:
        """Determine if column type should be changed."""
        if not sample_data:
            return None
        
        # For now, keep existing types (Phase 1 minimal implementation)
        # In later phases, we could implement more sophisticated type inference
        return None
    
    def generate_dsl_code(self, plan: TransformPlan) -> str:
        """Generate Transform DSL code from a plan."""
        dsl_lines = []
        dsl_lines.append("# Transform DSL Plan")
        dsl_lines.append(f"# Generated: {plan.metadata.get('auto_generated', False)}")
        dsl_lines.append(f"# Total operations: {len(plan.steps)}")
        dsl_lines.append("")
        
        for i, step in enumerate(plan.steps, 1):
            line = f"{i:2d}. {step.operation.value}({step.column}"
            if step.parameters:
                params = ", ".join(f"{k}={v}" for k, v in step.parameters.items())
                line += f", {params}"
            line += ")"
            dsl_lines.append(line)
        
        return "\n".join(dsl_lines)
