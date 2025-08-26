"""
Schema Inference Engine for Phase 1

Handles dialect detection, schema inference using Polars/PyArrow, and data type standardization.
"""

import logging
import os
import csv
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
import chardet
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    JSON = "json"
    JSONL = "jsonl"
    EXCEL = "excel"
    UNKNOWN = "unknown"


class DataType(str, Enum):
    """Standardized data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    JSON_OBJECT = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a column."""
    name: str
    original_name: str
    data_type: DataType
    nullable: bool
    unique_count: Optional[int] = None
    null_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: Optional[float] = None
    regex_patterns: Optional[List[str]] = None


@dataclass
class TableSchema:
    """Schema information for a table/file."""
    name: str
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    encoding: Optional[str] = None
    dialect: Optional[Dict[str, Any]] = None
    inferred_format: Optional[FileFormat] = None
    confidence: float = 0.0


class DialectDetector:
    """Detects file format and CSV dialect."""
    
    def __init__(self):
        self.max_sample_size = 8192  # 8KB sample for detection
    
    def detect_format(self, file_path: str) -> Tuple[FileFormat, float]:
        """Detect file format based on extension and content."""
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            # Extension-based detection
            if extension in ['.csv']:
                return FileFormat.CSV, 0.9
            elif extension in ['.tsv', '.tab']:
                return FileFormat.TSV, 0.9
            elif extension in ['.parquet', '.pq']:
                return FileFormat.PARQUET, 0.95
            elif extension in ['.json']:
                return FileFormat.JSON, 0.9
            elif extension in ['.jsonl', '.ndjson']:
                return FileFormat.JSONL, 0.9
            elif extension in ['.xlsx', '.xls']:
                return FileFormat.EXCEL, 0.9
            
            # Content-based detection
            if os.path.exists(file_path):
                return self._detect_by_content(file_path)
            
            return FileFormat.UNKNOWN, 0.0
            
        except Exception as e:
            logger.warning(f"Format detection failed: {e}")
            return FileFormat.UNKNOWN, 0.0
    
    def _detect_by_content(self, file_path: str) -> Tuple[FileFormat, float]:
        """Detect format by analyzing file content."""
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(self.max_sample_size)
            
            # Check for binary formats first
            if sample.startswith(b'PAR1'):
                return FileFormat.PARQUET, 0.95
            
            # Try to decode as text
            encoding_result = chardet.detect(sample)
            encoding = encoding_result.get('encoding', 'utf-8')
            
            try:
                text_sample = sample.decode(encoding)
            except UnicodeDecodeError:
                text_sample = sample.decode('utf-8', errors='ignore')
            
            # JSON detection
            text_sample_stripped = text_sample.strip()
            if text_sample_stripped.startswith(('{', '[')):
                try:
                    json.loads(text_sample_stripped)
                    return FileFormat.JSON, 0.8
                except json.JSONDecodeError:
                    pass
            
            # JSONL detection
            lines = text_sample_stripped.split('\n')[:5]
            jsonl_count = 0
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        jsonl_count += 1
                    except json.JSONDecodeError:
                        break
            
            if jsonl_count > 0 and jsonl_count == len([l for l in lines if l.strip()]):
                return FileFormat.JSONL, 0.8
            
            # CSV/TSV detection
            return self._detect_csv_dialect(text_sample)
            
        except Exception as e:
            logger.warning(f"Content-based detection failed: {e}")
            return FileFormat.UNKNOWN, 0.0
    
    def _detect_csv_dialect(self, sample: str) -> Tuple[FileFormat, float]:
        """Detect CSV dialect and format."""
        try:
            # Use csv.Sniffer to detect dialect
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            
            # Determine format based on delimiter
            if dialect.delimiter == '\t':
                return FileFormat.TSV, 0.8
            elif dialect.delimiter == ',':
                return FileFormat.CSV, 0.8
            else:
                # Other delimiter - treat as CSV
                return FileFormat.CSV, 0.6
                
        except Exception as e:
            logger.warning(f"CSV dialect detection failed: {e}")
            # Fallback: count delimiters
            comma_count = sample.count(',')
            tab_count = sample.count('\t')
            
            if tab_count > comma_count:
                return FileFormat.TSV, 0.5
            elif comma_count > 0:
                return FileFormat.CSV, 0.5
            
            return FileFormat.UNKNOWN, 0.0
    
    def detect_csv_dialect(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Detect CSV dialect parameters."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(self.max_sample_size)
            
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            
            return {
                "delimiter": dialect.delimiter,
                "quotechar": dialect.quotechar,
                "quoting": dialect.quoting,
                "skipinitialspace": dialect.skipinitialspace,
                "lineterminator": dialect.lineterminator,
                "has_header": sniffer.has_header(sample)
            }
            
        except Exception as e:
            logger.warning(f"CSV dialect detection failed: {e}")
            return None


class SchemaInferenceEngine:
    """Infers schema from various data sources using Polars/PyArrow."""
    
    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
        self.dialect_detector = DialectDetector()
    
    def infer_from_file(self, file_path: str, **kwargs) -> TableSchema:
        """Infer schema from a file."""
        try:
            logger.info(f"Starting schema inference for: {file_path}")
            
            # Detect format and dialect
            file_format, confidence = self.dialect_detector.detect_format(file_path)
            logger.info(f"Detected format: {file_format} (confidence: {confidence})")
            
            # Get file info
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Infer schema based on format
            if file_format == FileFormat.CSV:
                return self._infer_csv_schema(file_path, file_size, confidence)
            elif file_format == FileFormat.TSV:
                return self._infer_tsv_schema(file_path, file_size, confidence)
            elif file_format == FileFormat.PARQUET:
                return self._infer_parquet_schema(file_path, file_size, confidence)
            elif file_format == FileFormat.JSON:
                return self._infer_json_schema(file_path, file_size, confidence)
            elif file_format == FileFormat.JSONL:
                return self._infer_jsonl_schema(file_path, file_size, confidence)
            else:
                # Fallback to CSV
                logger.warning(f"Unknown format {file_format}, falling back to CSV")
                return self._infer_csv_schema(file_path, file_size, 0.3)
                
        except Exception as e:
            logger.error(f"Schema inference failed for {file_path}: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_csv_schema(self, file_path: str, file_size: int, confidence: float) -> TableSchema:
        """Infer schema from CSV file using Polars."""
        try:
            # Detect dialect
            dialect = self.dialect_detector.detect_csv_dialect(file_path)
            
            # Try to read with Polars
            try:
                # Read with automatic schema inference
                df = pl.read_csv(
                    file_path,
                    separator=dialect.get("delimiter", ",") if dialect else ",",
                    has_header=dialect.get("has_header", True) if dialect else True,
                    n_rows=self.sample_size,
                    infer_schema_length=min(1000, self.sample_size)
                )
                
                # Get encoding info
                encoding = self._detect_encoding(file_path)
                
                return self._create_schema_from_polars(
                    df, 
                    Path(file_path).stem,
                    file_size, 
                    encoding, 
                    dialect, 
                    FileFormat.CSV, 
                    confidence
                )
                
            except Exception as polars_error:
                logger.warning(f"Polars failed, trying pandas: {polars_error}")
                return self._infer_csv_with_pandas(file_path, file_size, confidence, dialect)
                
        except Exception as e:
            logger.error(f"CSV schema inference failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_tsv_schema(self, file_path: str, file_size: int, confidence: float) -> TableSchema:
        """Infer schema from TSV file."""
        try:
            df = pl.read_csv(
                file_path,
                separator="\t",
                has_header=True,
                n_rows=self.sample_size,
                infer_schema_length=min(1000, self.sample_size)
            )
            
            encoding = self._detect_encoding(file_path)
            dialect = {"delimiter": "\t", "has_header": True}
            
            return self._create_schema_from_polars(
                df, 
                Path(file_path).stem,
                file_size, 
                encoding, 
                dialect, 
                FileFormat.TSV, 
                confidence
            )
            
        except Exception as e:
            logger.error(f"TSV schema inference failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_parquet_schema(self, file_path: str, file_size: int, confidence: float) -> TableSchema:
        """Infer schema from Parquet file."""
        try:
            df = pl.read_parquet(file_path, n_rows=self.sample_size)
            
            return self._create_schema_from_polars(
                df, 
                Path(file_path).stem,
                file_size, 
                "binary", 
                None, 
                FileFormat.PARQUET, 
                confidence
            )
            
        except Exception as e:
            logger.error(f"Parquet schema inference failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_json_schema(self, file_path: str, file_size: int, confidence: float) -> TableSchema:
        """Infer schema from JSON file."""
        try:
            df = pl.read_json(file_path)
            if len(df) > self.sample_size:
                df = df.head(self.sample_size)
            
            encoding = self._detect_encoding(file_path)
            
            return self._create_schema_from_polars(
                df, 
                Path(file_path).stem,
                file_size, 
                encoding, 
                None, 
                FileFormat.JSON, 
                confidence
            )
            
        except Exception as e:
            logger.error(f"JSON schema inference failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_jsonl_schema(self, file_path: str, file_size: int, confidence: float) -> TableSchema:
        """Infer schema from JSONL file."""
        try:
            df = pl.read_ndjson(file_path, n_rows=self.sample_size)
            
            encoding = self._detect_encoding(file_path)
            
            return self._create_schema_from_polars(
                df, 
                Path(file_path).stem,
                file_size, 
                encoding, 
                None, 
                FileFormat.JSONL, 
                confidence
            )
            
        except Exception as e:
            logger.error(f"JSONL schema inference failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _infer_csv_with_pandas(self, file_path: str, file_size: int, confidence: float, dialect: Optional[Dict]) -> TableSchema:
        """Fallback CSV inference using pandas."""
        try:
            separator = dialect.get("delimiter", ",") if dialect else ","
            
            df = pd.read_csv(
                file_path,
                sep=separator,
                nrows=self.sample_size,
                low_memory=False
            )
            
            # Convert to Polars for consistency
            pl_df = pl.from_pandas(df)
            
            encoding = self._detect_encoding(file_path)
            
            return self._create_schema_from_polars(
                pl_df, 
                Path(file_path).stem,
                file_size, 
                encoding, 
                dialect, 
                FileFormat.CSV, 
                confidence * 0.8  # Lower confidence for fallback
            )
            
        except Exception as e:
            logger.error(f"Pandas CSV fallback failed: {e}")
            return self._create_error_schema(file_path, str(e))
    
    def _create_schema_from_polars(
        self, 
        df: pl.DataFrame, 
        table_name: str,
        file_size: int, 
        encoding: Optional[str], 
        dialect: Optional[Dict], 
        file_format: FileFormat, 
        confidence: float
    ) -> TableSchema:
        """Create TableSchema from Polars DataFrame."""
        columns = []
        
        for col_name in df.columns:
            col_series = df[col_name]
            
            # Get basic info
            original_name = col_name
            standardized_name = self._standardize_column_name(col_name)
            data_type = self._map_polars_type_to_standard(col_series.dtype)
            
            # Get statistics
            null_count = col_series.null_count()
            nullable = null_count > 0
            
            # Try to get unique count (may fail for complex types)
            try:
                unique_count = col_series.n_unique()
            except:
                unique_count = None
            
            # Get sample values (first 5 non-null)
            try:
                sample_values = col_series.drop_nulls().head(5).to_list()
            except:
                sample_values = []
            
            # Get min/max for numeric types
            min_value = None
            max_value = None
            try:
                if data_type in [DataType.INTEGER, DataType.FLOAT]:
                    min_value = col_series.min()
                    max_value = col_series.max()
            except:
                pass
            
            # Average length for strings
            avg_length = None
            try:
                if data_type == DataType.STRING:
                    avg_length = col_series.str.n_chars().mean()
            except:
                pass
            
            column_info = ColumnInfo(
                name=standardized_name,
                original_name=original_name,
                data_type=data_type,
                nullable=nullable,
                unique_count=unique_count,
                null_count=null_count,
                sample_values=sample_values,
                min_value=min_value,
                max_value=max_value,
                avg_length=avg_length
            )
            columns.append(column_info)
        
        return TableSchema(
            name=table_name,
            columns=columns,
            row_count=len(df),
            file_size_bytes=file_size,
            encoding=encoding,
            dialect=dialect,
            inferred_format=file_format,
            confidence=confidence
        )
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize column name (Phase 1 basic implementation)."""
        # Remove special characters and convert to lowercase
        import re
        standardized = re.sub(r'[^\w\s]', '', name.lower())
        standardized = re.sub(r'\s+', '_', standardized.strip())
        return standardized or "unknown_column"
    
    def _map_polars_type_to_standard(self, polars_type) -> DataType:
        """Map Polars data type to our standard types."""
        type_str = str(polars_type)
        
        if 'Int' in type_str or 'UInt' in type_str:
            return DataType.INTEGER
        elif 'Float' in type_str or 'Decimal' in type_str:
            return DataType.FLOAT
        elif 'Bool' in type_str:
            return DataType.BOOLEAN
        elif 'Date' in type_str and 'time' not in type_str.lower():
            return DataType.DATE
        elif 'Datetime' in type_str or 'Timestamp' in type_str:
            return DataType.DATETIME
        elif 'Time' in type_str:
            return DataType.TIME
        elif 'Utf8' in type_str or 'String' in type_str:
            return DataType.STRING
        elif 'Binary' in type_str:
            return DataType.BINARY
        else:
            return DataType.STRING  # Default fallback
    
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(8192)
            
            result = chardet.detect(sample)
            return result.get('encoding', 'utf-8')
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return 'utf-8'
    
    def _create_error_schema(self, file_path: str, error_msg: str) -> TableSchema:
        """Create error schema when inference fails."""
        return TableSchema(
            name=Path(file_path).stem,
            columns=[],
            row_count=0,
            file_size_bytes=0,
            encoding=None,
            dialect=None,
            inferred_format=FileFormat.UNKNOWN,
            confidence=0.0
        )
