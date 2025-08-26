"""
Real SQL Database Introspection for EnMapper

This module provides comprehensive database introspection capabilities,
including schema discovery, table metadata, column information, and relationships.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import polars as pl

# Database adapters
import sqlite3
import pymysql
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class ColumnType(str, Enum):
    """Standardized column types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    JSON = "json"
    BLOB = "blob"
    UUID = "uuid"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    normalized_type: ColumnType
    is_nullable: bool
    default_value: Optional[str]
    is_primary_key: bool
    is_foreign_key: bool
    max_length: Optional[int]
    precision: Optional[int]
    scale: Optional[int]
    foreign_key_references: Optional[Tuple[str, str]]  # (table, column)
    column_comment: Optional[str]
    ordinal_position: int


@dataclass
class IndexInfo:
    """Information about a database index."""
    name: str
    table_name: str
    column_names: List[str]
    is_unique: bool
    is_primary: bool
    index_type: Optional[str]


@dataclass
class ForeignKeyInfo:
    """Information about foreign key relationships."""
    constraint_name: str
    table_name: str
    column_name: str
    referenced_table: str
    referenced_column: str
    on_delete: Optional[str]
    on_update: Optional[str]


@dataclass
class TableInfo:
    """Comprehensive information about a database table."""
    name: str
    schema: str
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyInfo]
    indexes: List[IndexInfo]
    row_count: Optional[int]
    table_comment: Optional[str]
    table_type: str  # TABLE, VIEW, etc.
    estimated_size_bytes: Optional[int]


@dataclass
class DatabaseSchema:
    """Complete database schema information."""
    database_name: str
    database_type: DatabaseType
    version: Optional[str]
    schemas: List[str]
    tables: List[TableInfo]
    views: List[TableInfo]
    total_tables: int
    total_columns: int
    introspection_timestamp: str
    connection_info: Dict[str, Any]


class DatabaseIntrospector:
    """Comprehensive database introspection engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def introspect_database(
        self,
        db_type: DatabaseType,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> DatabaseSchema:
        """Main introspection method that delegates to specific database handlers."""
        
        self.logger.info(f"🔍 Starting database introspection: {db_type} - {database}")
        
        try:
            if db_type == DatabaseType.POSTGRESQL:
                return await self._introspect_postgresql(host, port, database, username, password, schema, **kwargs)
            elif db_type == DatabaseType.MYSQL:
                return await self._introspect_mysql(host, port, database, username, password, **kwargs)
            elif db_type == DatabaseType.SQLITE:
                return await self._introspect_sqlite(database, **kwargs)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Database introspection failed: {e}")
            raise
    
    async def _introspect_postgresql(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> DatabaseSchema:
        """Introspect PostgreSQL database."""
        
        connection = None
        try:
            # Connect to PostgreSQL
            connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                cursor_factory=RealDictCursor
            )
            
            cursor = connection.cursor()
            
            # Get database version
            cursor.execute("SELECT version()")
            version_info = cursor.fetchone()['version']
            
            # Get schemas
            cursor.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                ORDER BY schema_name
            """)
            schemas = [row['schema_name'] for row in cursor.fetchall()]
            
            # Focus on specific schema if provided
            target_schemas = [schema] if schema else schemas[:3]  # Limit to first 3 schemas
            
            tables = []
            views = []
            
            for target_schema in target_schemas:
                # Get tables and views
                schema_tables = await self._get_postgresql_tables(cursor, target_schema)
                schema_views = await self._get_postgresql_views(cursor, target_schema)
                
                tables.extend(schema_tables)
                views.extend(schema_views)
            
            total_columns = sum(len(table.columns) for table in tables + views)
            
            return DatabaseSchema(
                database_name=database,
                database_type=DatabaseType.POSTGRESQL,
                version=version_info,
                schemas=schemas,
                tables=tables,
                views=views,
                total_tables=len(tables),
                total_columns=total_columns,
                introspection_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                connection_info={
                    'host': host,
                    'port': port,
                    'database': database,
                    'username': username,
                    'schemas_introspected': target_schemas
                }
            )
            
        finally:
            if connection:
                connection.close()
    
    async def _get_postgresql_tables(self, cursor, schema: str) -> List[TableInfo]:
        """Get detailed table information from PostgreSQL."""
        tables = []
        
        # Get all tables in schema
        cursor.execute("""
            SELECT table_name, table_type, table_comment
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """, (schema,))
        
        table_rows = cursor.fetchall()
        
        for table_row in table_rows:
            table_name = table_row['table_name']
            
            # Get columns
            columns = await self._get_postgresql_columns(cursor, schema, table_name)
            
            # Get primary keys
            primary_keys = await self._get_postgresql_primary_keys(cursor, schema, table_name)
            
            # Get foreign keys
            foreign_keys = await self._get_postgresql_foreign_keys(cursor, schema, table_name)
            
            # Get indexes
            indexes = await self._get_postgresql_indexes(cursor, schema, table_name)
            
            # Get row count (with limit for performance)
            try:
                cursor.execute(f'SELECT COUNT(*) as count FROM "{schema}"."{table_name}" LIMIT 100000')
                row_count = cursor.fetchone()['count']
            except Exception:
                row_count = None
            
            # Get table size estimate
            try:
                cursor.execute("""
                    SELECT pg_total_relation_size(%s) as size_bytes
                """, (f'"{schema}"."{table_name}"',))
                size_result = cursor.fetchone()
                estimated_size = size_result['size_bytes'] if size_result else None
            except Exception:
                estimated_size = None
            
            tables.append(TableInfo(
                name=table_name,
                schema=schema,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=row_count,
                table_comment=table_row.get('table_comment'),
                table_type=table_row['table_type'],
                estimated_size_bytes=estimated_size
            ))
        
        return tables
    
    async def _get_postgresql_columns(self, cursor, schema: str, table: str) -> List[ColumnInfo]:
        """Get detailed column information from PostgreSQL."""
        cursor.execute("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.ordinal_position,
                col_description(pgc.oid, c.ordinal_position) as column_comment,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
                fk.foreign_table_name,
                fk.foreign_column_name
            FROM information_schema.columns c
            LEFT JOIN pg_class pgc ON pgc.relname = c.table_name
            LEFT JOIN pg_namespace pgn ON pgn.oid = pgc.relnamespace AND pgn.nspname = c.table_schema
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name 
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY' 
                    AND tc.table_schema = %s 
                    AND tc.table_name = %s
            ) pk ON pk.column_name = c.column_name
            LEFT JOIN (
                SELECT 
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu 
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                    AND tc.table_schema = %s 
                    AND tc.table_name = %s
            ) fk ON fk.column_name = c.column_name
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position
        """, (schema, table, schema, table, schema, table))
        
        columns = []
        for row in cursor.fetchall():
            normalized_type = self._normalize_postgresql_type(row['data_type'])
            
            foreign_key_ref = None
            if row['is_foreign_key'] and row['foreign_table_name']:
                foreign_key_ref = (row['foreign_table_name'], row['foreign_column_name'])
            
            columns.append(ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                normalized_type=normalized_type,
                is_nullable=row['is_nullable'] == 'YES',
                default_value=row['column_default'],
                is_primary_key=row['is_primary_key'],
                is_foreign_key=row['is_foreign_key'],
                max_length=row['character_maximum_length'],
                precision=row['numeric_precision'],
                scale=row['numeric_scale'],
                foreign_key_references=foreign_key_ref,
                column_comment=row['column_comment'],
                ordinal_position=row['ordinal_position']
            ))
        
        return columns
    
    async def _get_postgresql_primary_keys(self, cursor, schema: str, table: str) -> List[str]:
        """Get primary key columns from PostgreSQL."""
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name 
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY' 
                AND tc.table_schema = %s 
                AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
        """, (schema, table))
        
        return [row['column_name'] for row in cursor.fetchall()]
    
    async def _get_postgresql_foreign_keys(self, cursor, schema: str, table: str) -> List[ForeignKeyInfo]:
        """Get foreign key information from PostgreSQL."""
        cursor.execute("""
            SELECT 
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                rc.delete_rule,
                rc.update_rule
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu 
                ON ccu.constraint_name = tc.constraint_name
            JOIN information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_schema = %s 
                AND tc.table_name = %s
        """, (schema, table))
        
        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append(ForeignKeyInfo(
                constraint_name=row['constraint_name'],
                table_name=table,
                column_name=row['column_name'],
                referenced_table=row['foreign_table_name'],
                referenced_column=row['foreign_column_name'],
                on_delete=row['delete_rule'],
                on_update=row['update_rule']
            ))
        
        return foreign_keys
    
    async def _get_postgresql_indexes(self, cursor, schema: str, table: str) -> List[IndexInfo]:
        """Get index information from PostgreSQL."""
        cursor.execute("""
            SELECT 
                i.relname as index_name,
                am.amname as index_type,
                ix.indisunique,
                ix.indisprimary,
                array_agg(a.attname ORDER BY c.ordinality) as column_names
            FROM pg_class t
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON i.relam = am.oid
            JOIN unnest(ix.indkey) WITH ORDINALITY c(colnum, ordinality) ON true
            JOIN pg_attribute a ON t.oid = a.attrelid AND a.attnum = c.colnum
            WHERE n.nspname = %s AND t.relname = %s
            GROUP BY i.relname, am.amname, ix.indisunique, ix.indisprimary
        """, (schema, table))
        
        indexes = []
        for row in cursor.fetchall():
            indexes.append(IndexInfo(
                name=row['index_name'],
                table_name=table,
                column_names=row['column_names'],
                is_unique=row['indisunique'],
                is_primary=row['indisprimary'],
                index_type=row['index_type']
            ))
        
        return indexes
    
    async def _get_postgresql_views(self, cursor, schema: str) -> List[TableInfo]:
        """Get view information from PostgreSQL."""
        cursor.execute("""
            SELECT table_name, view_definition
            FROM information_schema.views 
            WHERE table_schema = %s
            ORDER BY table_name
        """, (schema,))
        
        views = []
        for row in cursor.fetchall():
            view_name = row['table_name']
            
            # Get view columns
            columns = await self._get_postgresql_columns(cursor, schema, view_name)
            
            views.append(TableInfo(
                name=view_name,
                schema=schema,
                columns=columns,
                primary_keys=[],
                foreign_keys=[],
                indexes=[],
                row_count=None,
                table_comment=None,
                table_type='VIEW',
                estimated_size_bytes=None
            ))
        
        return views
    
    def _normalize_postgresql_type(self, pg_type: str) -> ColumnType:
        """Normalize PostgreSQL data types to standard types."""
        pg_type = pg_type.lower()
        
        if 'char' in pg_type or 'text' in pg_type:
            return ColumnType.STRING
        elif 'int' in pg_type or 'serial' in pg_type:
            return ColumnType.INTEGER
        elif 'float' in pg_type or 'real' in pg_type or 'double' in pg_type:
            return ColumnType.FLOAT
        elif 'numeric' in pg_type or 'decimal' in pg_type:
            return ColumnType.DECIMAL
        elif 'bool' in pg_type:
            return ColumnType.BOOLEAN
        elif 'date' in pg_type and 'time' not in pg_type:
            return ColumnType.DATE
        elif 'timestamp' in pg_type or 'datetime' in pg_type:
            return ColumnType.DATETIME
        elif 'time' in pg_type:
            return ColumnType.TIME
        elif 'json' in pg_type:
            return ColumnType.JSON
        elif 'uuid' in pg_type:
            return ColumnType.UUID
        elif 'bytea' in pg_type or 'blob' in pg_type:
            return ColumnType.BLOB
        else:
            return ColumnType.UNKNOWN
    
    async def _introspect_mysql(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        **kwargs
    ) -> DatabaseSchema:
        """Introspect MySQL database."""
        
        connection = None
        try:
            # Connect to MySQL
            connection = pymysql.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            cursor = connection.cursor()
            
            # Get MySQL version
            cursor.execute("SELECT VERSION() as version")
            version_info = cursor.fetchone()['version']
            
            # Get database list (schemas)
            cursor.execute("SHOW DATABASES")
            all_schemas = [row['Database'] for row in cursor.fetchall()]
            
            # Get tables from current database
            tables = await self._get_mysql_tables(cursor, database)
            
            return DatabaseSchema(
                database_name=database,
                database_type=DatabaseType.MYSQL,
                version=version_info,
                schemas=[database],  # MySQL uses database as schema
                tables=tables,
                views=[],  # Views would require additional logic
                total_tables=len(tables),
                total_columns=sum(len(table.columns) for table in tables),
                introspection_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                connection_info={
                    'host': host,
                    'port': port,
                    'database': database,
                    'username': username,
                    'available_databases': all_schemas
                }
            )
            
        finally:
            if connection:
                connection.close()
    
    async def _get_mysql_tables(self, cursor, database: str) -> List[TableInfo]:
        """Get detailed table information from MySQL."""
        tables = []
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        table_rows = cursor.fetchall()
        
        for table_row in table_rows:
            table_name = list(table_row.values())[0]  # MySQL returns table name as value
            
            # Get columns
            columns = await self._get_mysql_columns(cursor, database, table_name)
            
            # Get primary keys
            primary_keys = await self._get_mysql_primary_keys(cursor, table_name)
            
            # Get row count
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}` LIMIT 100000")
                row_count = cursor.fetchone()['count']
            except Exception:
                row_count = None
            
            tables.append(TableInfo(
                name=table_name,
                schema=database,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=[],  # Would require additional queries
                indexes=[],       # Would require additional queries
                row_count=row_count,
                table_comment=None,
                table_type='TABLE',
                estimated_size_bytes=None
            ))
        
        return tables
    
    async def _get_mysql_columns(self, cursor, database: str, table: str) -> List[ColumnInfo]:
        """Get detailed column information from MySQL."""
        cursor.execute(f"DESCRIBE `{table}`")
        
        columns = []
        for i, row in enumerate(cursor.fetchall()):
            normalized_type = self._normalize_mysql_type(row['Type'])
            
            columns.append(ColumnInfo(
                name=row['Field'],
                data_type=row['Type'],
                normalized_type=normalized_type,
                is_nullable=row['Null'] == 'YES',
                default_value=row['Default'],
                is_primary_key=row['Key'] == 'PRI',
                is_foreign_key=row['Key'] == 'MUL',
                max_length=None,  # Would need parsing of Type field
                precision=None,
                scale=None,
                foreign_key_references=None,
                column_comment=None,
                ordinal_position=i + 1
            ))
        
        return columns
    
    async def _get_mysql_primary_keys(self, cursor, table: str) -> List[str]:
        """Get primary key columns from MySQL."""
        cursor.execute(f"SHOW KEYS FROM `{table}` WHERE Key_name = 'PRIMARY'")
        return [row['Column_name'] for row in cursor.fetchall()]
    
    def _normalize_mysql_type(self, mysql_type: str) -> ColumnType:
        """Normalize MySQL data types to standard types."""
        mysql_type = mysql_type.lower()
        
        if 'char' in mysql_type or 'text' in mysql_type:
            return ColumnType.STRING
        elif 'int' in mysql_type:
            return ColumnType.INTEGER
        elif 'float' in mysql_type or 'double' in mysql_type:
            return ColumnType.FLOAT
        elif 'decimal' in mysql_type or 'numeric' in mysql_type:
            return ColumnType.DECIMAL
        elif 'bool' in mysql_type or 'bit' in mysql_type:
            return ColumnType.BOOLEAN
        elif mysql_type == 'date':
            return ColumnType.DATE
        elif 'datetime' in mysql_type or 'timestamp' in mysql_type:
            return ColumnType.DATETIME
        elif 'time' in mysql_type:
            return ColumnType.TIME
        elif 'json' in mysql_type:
            return ColumnType.JSON
        elif 'blob' in mysql_type or 'binary' in mysql_type:
            return ColumnType.BLOB
        else:
            return ColumnType.UNKNOWN
    
    async def _introspect_sqlite(self, database_path: str, **kwargs) -> DatabaseSchema:
        """Introspect SQLite database."""
        
        connection = None
        try:
            # Connect to SQLite
            connection = sqlite3.connect(database_path)
            connection.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = connection.cursor()
            
            # Get SQLite version
            cursor.execute("SELECT sqlite_version() as version")
            version_info = cursor.fetchone()['version']
            
            # Get all tables
            tables = await self._get_sqlite_tables(cursor)
            
            return DatabaseSchema(
                database_name=database_path,
                database_type=DatabaseType.SQLITE,
                version=version_info,
                schemas=['main'],  # SQLite uses 'main' schema
                tables=tables,
                views=[],  # Views would require additional logic
                total_tables=len(tables),
                total_columns=sum(len(table.columns) for table in tables),
                introspection_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                connection_info={
                    'database_path': database_path
                }
            )
            
        finally:
            if connection:
                connection.close()
    
    async def _get_sqlite_tables(self, cursor) -> List[TableInfo]:
        """Get detailed table information from SQLite."""
        tables = []
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table_rows = cursor.fetchall()
        
        for table_row in table_rows:
            table_name = table_row['name']
            
            # Get columns using PRAGMA table_info
            columns = await self._get_sqlite_columns(cursor, table_name)
            
            # Get primary keys
            primary_keys = [col.name for col in columns if col.is_primary_key]
            
            # Get row count
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}` LIMIT 100000")
                row_count = cursor.fetchone()['count']
            except Exception:
                row_count = None
            
            tables.append(TableInfo(
                name=table_name,
                schema='main',
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=[],  # Would require additional queries
                indexes=[],       # Would require additional queries
                row_count=row_count,
                table_comment=None,
                table_type='TABLE',
                estimated_size_bytes=None
            ))
        
        return tables
    
    async def _get_sqlite_columns(self, cursor, table: str) -> List[ColumnInfo]:
        """Get detailed column information from SQLite."""
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        
        columns = []
        for row in cursor.fetchall():
            normalized_type = self._normalize_sqlite_type(row['type'])
            
            columns.append(ColumnInfo(
                name=row['name'],
                data_type=row['type'],
                normalized_type=normalized_type,
                is_nullable=not row['notnull'],
                default_value=row['dflt_value'],
                is_primary_key=bool(row['pk']),
                is_foreign_key=False,  # Would require additional queries
                max_length=None,
                precision=None,
                scale=None,
                foreign_key_references=None,
                column_comment=None,
                ordinal_position=row['cid']
            ))
        
        return columns
    
    def _normalize_sqlite_type(self, sqlite_type: str) -> ColumnType:
        """Normalize SQLite data types to standard types."""
        if not sqlite_type:
            return ColumnType.UNKNOWN
        
        sqlite_type = sqlite_type.lower()
        
        if 'text' in sqlite_type or 'char' in sqlite_type:
            return ColumnType.STRING
        elif 'int' in sqlite_type:
            return ColumnType.INTEGER
        elif 'real' in sqlite_type or 'float' in sqlite_type or 'double' in sqlite_type:
            return ColumnType.FLOAT
        elif 'numeric' in sqlite_type or 'decimal' in sqlite_type:
            return ColumnType.DECIMAL
        elif 'bool' in sqlite_type:
            return ColumnType.BOOLEAN
        elif 'date' in sqlite_type:
            return ColumnType.DATE
        elif 'time' in sqlite_type:
            return ColumnType.DATETIME
        elif 'blob' in sqlite_type:
            return ColumnType.BLOB
        else:
            return ColumnType.UNKNOWN


# Global introspector instance
database_introspector = DatabaseIntrospector()
