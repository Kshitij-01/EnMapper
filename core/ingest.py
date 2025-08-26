"""
LLM-INGEST Agent for Phase 1

Handles data source ingestion, schema inference, and standardization for both files and SQL sources.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlglot
from sqlglot import parse_one, transpile
from sqlglot.errors import ParseError

from .inference import SchemaInferenceEngine, TableSchema, ColumnInfo, DataType
from .database import DatabaseConnectionTester, DatabaseType
# from .database import DatabaseIntrospector  # Not implemented yet
from settings import DatabaseSettings

logger = logging.getLogger(__name__)


class IngestStatus(str, Enum):
    """Status of ingestion process."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestResult:
    """Result of data ingestion."""
    status: IngestStatus
    schema: Optional[TableSchema] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None


@dataclass
class StandardCatalog:
    """Standardized schema catalog."""
    tables: List[TableSchema]
    source_type: str
    source_location: str
    created_at: str
    version: str = "1.0"
    total_tables: int = 0
    total_columns: int = 0
    confidence_score: float = 0.0


class LLMIngestAgent:
    """
    LLM-INGEST Agent for processing data sources.
    
    Responsibilities:
    - Dialect detection and schema inference
    - SQL introspection using SQLGlot
    - Data standardization and cataloging
    - Sample generation with PII considerations
    """
    
    def __init__(self, database_settings: Optional[DatabaseSettings] = None):
        self.schema_engine = SchemaInferenceEngine()
        self.database_settings = database_settings
        self.db_tester = DatabaseConnectionTester(database_settings) if database_settings else None
        # self.db_introspector = DatabaseIntrospector(self.db_tester) if self.db_tester else None  # Not implemented yet
    
    async def ingest_file(self, file_path: str, **kwargs) -> IngestResult:
        """
        Ingest data from a file source.
        
        Args:
            file_path: Path to the file to ingest
            **kwargs: Additional options (sample_size, etc.)
        
        Returns:
            IngestResult with schema and sample data
        """
        try:
            import time
            start_time = time.time()
            
            logger.info(f"Starting file ingestion: {file_path}")
            
            # Infer schema using the inference engine
            schema = self.schema_engine.infer_from_file(file_path, **kwargs)
            
            # Generate sample data if requested
            sample_data = None
            sample_size = kwargs.get('sample_size', 10)
            if sample_size > 0 and schema.row_count and schema.row_count > 0:
                sample_data = await self._generate_sample_data(file_path, schema, sample_size)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create metadata
            metadata = {
                "source_type": "file",
                "source_path": file_path,
                "inferred_format": schema.inferred_format.value if schema.inferred_format else "unknown",
                "encoding": schema.encoding,
                "dialect": schema.dialect,
                "confidence": schema.confidence,
                "processing_time_ms": processing_time
            }
            
            logger.info(f"File ingestion completed: {len(schema.columns)} columns, {schema.row_count} rows")
            
            return IngestResult(
                status=IngestStatus.COMPLETED,
                schema=schema,
                sample_data=sample_data,
                metadata=metadata,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            return IngestResult(
                status=IngestStatus.FAILED,
                error_message=str(e)
            )
    
    async def ingest_sql(self, connection_params: Dict[str, Any], **kwargs) -> IngestResult:
        """
        Ingest data from a SQL source.
        
        Args:
            connection_params: Database connection parameters
            **kwargs: Additional options (tables, query, etc.)
        
        Returns:
            IngestResult with schema information
        """
        try:
            import time
            start_time = time.time()
            
            logger.info("Starting SQL ingestion")
            
            # Determine database type
            db_type_str = connection_params.get("database_type", "postgresql")
            db_type = DatabaseType(db_type_str.lower())
            
            # Test connection first
            if self.db_tester:
                if db_type == DatabaseType.MYSQL:
                    connection_result = await self.db_tester.test_mysql()
                elif db_type == DatabaseType.SQLITE:
                    connection_result = await self.db_tester.test_sqlite()
                else:
                    connection_result = await self.db_tester.test_postgresql()
                
                if not connection_result.success:
                    raise Exception(f"Database connection failed: {connection_result.error_message}")
            
            # Get schema information (simplified for Phase 1)
            # if self.db_introspector:
            #     schema_info = await self.db_introspector.get_schema_info(db_type, **connection_params)
            # else:
            schema_info = {"tables": [], "database_type": db_type_str}
            
            # Convert to our TableSchema format
            tables = []
            for table_info in schema_info.get("tables", []):
                columns = []
                for col_info in table_info.get("columns", []):
                    column = ColumnInfo(
                        name=self._standardize_column_name(col_info["name"]),
                        original_name=col_info["name"],
                        data_type=self._map_sql_type_to_standard(col_info["type"]),
                        nullable=col_info.get("nullable", True),
                        sample_values=None  # SQL introspection doesn't provide samples
                    )
                    columns.append(column)
                
                table_schema = TableSchema(
                    name=f"{table_info.get('schema', 'public')}.{table_info['name']}",
                    columns=columns,
                    row_count=None,  # Would need actual query to get row count
                    inferred_format=None,
                    confidence=0.9  # High confidence for SQL schema
                )
                tables.append(table_schema)
            
            processing_time = (time.time() - start_time) * 1000
            
            # For now, return the first table or create a summary
            primary_schema = tables[0] if tables else TableSchema(
                name="unknown",
                columns=[],
                confidence=0.0
            )
            
            metadata = {
                "source_type": "sql",
                "database_type": db_type_str,
                "total_tables": len(tables),
                "table_names": [t.name for t in tables],
                "processing_time_ms": processing_time
            }
            
            logger.info(f"SQL ingestion completed: {len(tables)} tables")
            
            return IngestResult(
                status=IngestStatus.COMPLETED,
                schema=primary_schema,
                metadata=metadata,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"SQL ingestion failed: {e}")
            return IngestResult(
                status=IngestStatus.FAILED,
                error_message=str(e)
            )
    
    async def create_standard_catalog(self, ingest_results: List[IngestResult]) -> StandardCatalog:
        """
        Create a standardized catalog from ingestion results.
        
        Args:
            ingest_results: List of IngestResult objects
        
        Returns:
            StandardCatalog with consolidated schema information
        """
        try:
            from datetime import datetime
            
            tables = []
            total_confidence = 0.0
            successful_results = 0
            
            for result in ingest_results:
                if result.status == IngestStatus.COMPLETED and result.schema:
                    tables.append(result.schema)
                    total_confidence += result.schema.confidence
                    successful_results += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / successful_results if successful_results > 0 else 0.0
            
            # Determine source info
            source_type = "mixed"
            source_location = "multiple"
            if len(ingest_results) == 1 and ingest_results[0].metadata:
                metadata = ingest_results[0].metadata
                source_type = metadata.get("source_type", "unknown")
                source_location = metadata.get("source_path") or metadata.get("database_type", "unknown")
            
            catalog = StandardCatalog(
                tables=tables,
                source_type=source_type,
                source_location=source_location,
                created_at=datetime.utcnow().isoformat(),
                total_tables=len(tables),
                total_columns=sum(len(t.columns) for t in tables),
                confidence_score=avg_confidence
            )
            
            logger.info(f"Created standard catalog: {catalog.total_tables} tables, {catalog.total_columns} columns")
            
            return catalog
            
        except Exception as e:
            logger.error(f"Failed to create standard catalog: {e}")
            return StandardCatalog(
                tables=[],
                source_type="error",
                source_location="unknown",
                created_at="",
                confidence_score=0.0
            )
    
    async def _generate_sample_data(self, file_path: str, schema: TableSchema, sample_size: int) -> List[Dict[str, Any]]:
        """Generate sample data from file (simplified for Phase 1)."""
        try:
            import polars as pl
            
            # Read sample rows
            if schema.inferred_format and schema.inferred_format.value == "parquet":
                df = pl.read_parquet(file_path, n_rows=sample_size)
            elif schema.inferred_format and schema.inferred_format.value in ["json"]:
                df = pl.read_json(file_path)
                if len(df) > sample_size:
                    df = df.head(sample_size)
            elif schema.inferred_format and schema.inferred_format.value in ["jsonl"]:
                df = pl.read_ndjson(file_path, n_rows=sample_size)
            else:
                # Default to CSV
                delimiter = ","
                if schema.dialect and "delimiter" in schema.dialect:
                    delimiter = schema.dialect["delimiter"]
                
                df = pl.read_csv(
                    file_path,
                    separator=delimiter,
                    n_rows=sample_size,
                    has_header=schema.dialect.get("has_header", True) if schema.dialect else True
                )
            
            # Convert to list of dicts
            sample_data = df.to_dicts()
            
            logger.info(f"Generated {len(sample_data)} sample records")
            return sample_data
            
        except Exception as e:
            logger.warning(f"Sample data generation failed: {e}")
            return []
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize column name."""
        import re
        standardized = re.sub(r'[^\w\s]', '', name.lower())
        standardized = re.sub(r'\s+', '_', standardized.strip())
        return standardized or "unknown_column"
    
    def _map_sql_type_to_standard(self, sql_type: str) -> DataType:
        """Map SQL data type to our standard types."""
        sql_type_lower = sql_type.lower()
        
        if any(t in sql_type_lower for t in ['int', 'serial', 'bigint', 'smallint']):
            return DataType.INTEGER
        elif any(t in sql_type_lower for t in ['float', 'double', 'decimal', 'numeric', 'real']):
            return DataType.FLOAT
        elif any(t in sql_type_lower for t in ['bool', 'boolean']):
            return DataType.BOOLEAN
        elif any(t in sql_type_lower for t in ['date']) and 'time' not in sql_type_lower:
            return DataType.DATE
        elif any(t in sql_type_lower for t in ['timestamp', 'datetime']):
            return DataType.DATETIME
        elif any(t in sql_type_lower for t in ['time']):
            return DataType.TIME
        elif any(t in sql_type_lower for t in ['text', 'varchar', 'char', 'string']):
            return DataType.STRING
        elif any(t in sql_type_lower for t in ['json', 'jsonb']):
            return DataType.JSON_OBJECT
        elif any(t in sql_type_lower for t in ['blob', 'binary', 'bytea']):
            return DataType.BINARY
        else:
            return DataType.STRING  # Default fallback


class SQLQueryAnalyzer:
    """Analyzes SQL queries using SQLGlot."""
    
    def __init__(self):
        self.supported_dialects = [
            "mysql",
            "postgresql", 
            "sqlite",
            "sql server",
            "oracle",
            "bigquery",
            "snowflake"
        ]
    
    def analyze_query(self, query: str, dialect: str = "sql") -> Dict[str, Any]:
        """
        Analyze a SQL query and extract metadata.
        
        Args:
            query: SQL query string
            dialect: SQL dialect (mysql, postgresql, etc.)
        
        Returns:
            Dictionary with query analysis results
        """
        try:
            logger.info(f"Analyzing SQL query with dialect: {dialect}")
            
            # Parse the query
            parsed = parse_one(query, dialect=dialect)
            
            # Extract information
            analysis = {
                "query_type": self._get_query_type(parsed),
                "tables": self._extract_tables(parsed),
                "columns": self._extract_columns(parsed),
                "joins": self._extract_joins(parsed),
                "where_conditions": self._extract_where_conditions(parsed),
                "aggregations": self._extract_aggregations(parsed),
                "is_read_only": self._is_read_only(parsed),
                "complexity_score": self._calculate_complexity(parsed),
                "parsed_sql": str(parsed)
            }
            
            logger.info(f"Query analysis completed: {analysis['query_type']} query with {len(analysis['tables'])} tables")
            
            return analysis
            
        except ParseError as e:
            logger.error(f"SQL parsing failed: {e}")
            return {
                "error": "Parse error",
                "error_message": str(e),
                "query_type": "unknown",
                "tables": [],
                "columns": [],
                "is_read_only": False,
                "complexity_score": 0
            }
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "error": "Analysis error",
                "error_message": str(e),
                "query_type": "unknown",
                "tables": [],
                "columns": [],
                "is_read_only": False,
                "complexity_score": 0
            }
    
    def _get_query_type(self, parsed) -> str:
        """Determine the type of SQL query."""
        if hasattr(parsed, 'sql_name'):
            return parsed.sql_name().lower()
        return "unknown"
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from the query."""
        tables = []
        try:
            for table in parsed.find_all(sqlglot.expressions.Table):
                if table.name:
                    tables.append(table.name)
        except:
            pass
        return list(set(tables))  # Remove duplicates
    
    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from the query."""
        columns = []
        try:
            for column in parsed.find_all(sqlglot.expressions.Column):
                if column.name:
                    columns.append(column.name)
        except:
            pass
        return list(set(columns))  # Remove duplicates
    
    def _extract_joins(self, parsed) -> List[Dict[str, str]]:
        """Extract join information."""
        joins = []
        try:
            for join in parsed.find_all(sqlglot.expressions.Join):
                join_info = {
                    "type": join.side if hasattr(join, 'side') else "inner",
                    "table": str(join.this) if hasattr(join, 'this') else "unknown"
                }
                joins.append(join_info)
        except:
            pass
        return joins
    
    def _extract_where_conditions(self, parsed) -> List[str]:
        """Extract WHERE clause conditions."""
        conditions = []
        try:
            for where in parsed.find_all(sqlglot.expressions.Where):
                conditions.append(str(where.this))
        except:
            pass
        return conditions
    
    def _extract_aggregations(self, parsed) -> List[str]:
        """Extract aggregation functions."""
        aggregations = []
        try:
            for func in parsed.find_all(sqlglot.expressions.AggFunc):
                aggregations.append(func.sql_name())
        except:
            pass
        return list(set(aggregations))
    
    def _is_read_only(self, parsed) -> bool:
        """Check if the query is read-only (SELECT)."""
        query_type = self._get_query_type(parsed)
        return query_type.lower() in ['select']
    
    def _calculate_complexity(self, parsed) -> int:
        """Calculate a complexity score for the query."""
        complexity = 0
        try:
            # Count various elements
            complexity += len(list(parsed.find_all(sqlglot.expressions.Table)))
            complexity += len(list(parsed.find_all(sqlglot.expressions.Join))) * 2
            complexity += len(list(parsed.find_all(sqlglot.expressions.Subquery))) * 3
            complexity += len(list(parsed.find_all(sqlglot.expressions.AggFunc)))
            complexity += len(list(parsed.find_all(sqlglot.expressions.Where)))
        except:
            pass
        return complexity
