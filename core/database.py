"""
Database connection utilities and testing for multiple database types.

Supports PostgreSQL, MySQL, and SQLite connections with validation and testing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import os

# Database drivers
import aiosqlite
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
import mysql.connector
import pymysql

from settings import DatabaseSettings

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class ConnectionResult:
    """Result of a database connection test."""
    success: bool
    database_type: DatabaseType
    connection_info: Dict[str, Any]
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    server_version: Optional[str] = None
    schema_info: Optional[Dict[str, Any]] = None


class DatabaseConnectionTester:
    """Test connections to various database types."""
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
    
    async def test_all_connections(self) -> Dict[str, ConnectionResult]:
        """Test all configured database connections."""
        results = {}
        
        # Test PostgreSQL (main database)
        try:
            results["postgresql"] = await self.test_postgresql()
        except Exception as e:
            logger.error(f"PostgreSQL test failed: {e}")
            results["postgresql"] = ConnectionResult(
                success=False,
                database_type=DatabaseType.POSTGRESQL,
                connection_info={"url": self.settings.postgres_url},
                error_message=str(e)
            )
        
        # Test MySQL
        try:
            results["mysql"] = await self.test_mysql()
        except Exception as e:
            logger.error(f"MySQL test failed: {e}")
            results["mysql"] = ConnectionResult(
                success=False,
                database_type=DatabaseType.MYSQL,
                connection_info={
                    "host": self.settings.mysql_host,
                    "port": self.settings.mysql_port,
                    "database": self.settings.mysql_database
                },
                error_message=str(e)
            )
        
        # Test SQLite
        try:
            results["sqlite"] = await self.test_sqlite()
        except Exception as e:
            logger.error(f"SQLite test failed: {e}")
            results["sqlite"] = ConnectionResult(
                success=False,
                database_type=DatabaseType.SQLITE,
                connection_info={"path": self.settings.sqlite_path},
                error_message=str(e)
            )
        
        return results
    
    async def test_postgresql(self) -> ConnectionResult:
        """Test PostgreSQL connection."""
        import time
        start_time = time.time()
        
        try:
            # Create async engine
            engine = create_async_engine(
                self.settings.postgres_url,
                pool_size=1,
                max_overflow=0,
                pool_timeout=self.settings.connection_timeout,
                echo=False
            )
            
            async with engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(sa.text("SELECT version()"))
                version = result.scalar()
                
                # Get schema info
                schema_result = await conn.execute(sa.text("""
                    SELECT 
                        schemaname,
                        COUNT(tablename) as table_count
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    GROUP BY schemaname
                """))
                
                schemas = [{"schema": row[0], "tables": row[1]} for row in schema_result.fetchall()]
                
                # Test PGVector extension if available
                try:
                    await conn.execute(sa.text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
                    pgvector_available = True
                except:
                    pgvector_available = False
            
            await engine.dispose()
            
            response_time = (time.time() - start_time) * 1000
            
            return ConnectionResult(
                success=True,
                database_type=DatabaseType.POSTGRESQL,
                connection_info={
                    "url": self.settings.postgres_url,
                    "pool_size": self.settings.postgres_pool_size
                },
                response_time_ms=response_time,
                server_version=version,
                schema_info={
                    "schemas": schemas,
                    "pgvector_available": pgvector_available
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ConnectionResult(
                success=False,
                database_type=DatabaseType.POSTGRESQL,
                connection_info={"url": self.settings.postgres_url},
                error_message=str(e),
                response_time_ms=response_time
            )
    
    async def test_mysql(self) -> ConnectionResult:
        """Test MySQL connection."""
        import time
        start_time = time.time()
        
        try:
            # Build connection config
            config = {
                'host': self.settings.mysql_host,
                'port': self.settings.mysql_port,
                'user': self.settings.mysql_username,
                'database': self.settings.mysql_database,
                'charset': self.settings.mysql_charset,
                'connect_timeout': self.settings.connection_timeout,
                'autocommit': True
            }
            
            if self.settings.mysql_password:
                config['password'] = self.settings.mysql_password.get_secret_value()
            
            if self.settings.mysql_ssl_disabled:
                config['ssl_disabled'] = True
            
            # Test connection using mysql-connector-python
            connection = mysql.connector.connect(**config)
            cursor = connection.cursor()
            
            # Get server version
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            
            # Get schema info
            cursor.execute("""
                SELECT 
                    TABLE_SCHEMA,
                    COUNT(TABLE_NAME) as table_count
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
                GROUP BY TABLE_SCHEMA
            """)
            
            schemas = [{"schema": row[0], "tables": row[1]} for row in cursor.fetchall()]
            
            cursor.close()
            connection.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return ConnectionResult(
                success=True,
                database_type=DatabaseType.MYSQL,
                connection_info={
                    "host": self.settings.mysql_host,
                    "port": self.settings.mysql_port,
                    "database": self.settings.mysql_database
                },
                response_time_ms=response_time,
                server_version=version,
                schema_info={"schemas": schemas}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ConnectionResult(
                success=False,
                database_type=DatabaseType.MYSQL,
                connection_info={
                    "host": self.settings.mysql_host,
                    "port": self.settings.mysql_port,
                    "database": self.settings.mysql_database
                },
                error_message=str(e),
                response_time_ms=response_time
            )
    
    async def test_sqlite(self) -> ConnectionResult:
        """Test SQLite connection."""
        import time
        start_time = time.time()
        
        try:
            # Ensure directory exists
            db_path = self.settings.sqlite_path
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Test connection
            async with aiosqlite.connect(
                db_path,
                timeout=self.settings.sqlite_timeout,
                check_same_thread=self.settings.sqlite_check_same_thread
            ) as db:
                # Get SQLite version
                async with db.execute("SELECT sqlite_version()") as cursor:
                    version = await cursor.fetchone()
                    version = version[0] if version else "unknown"
                
                # Get table info
                async with db.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """) as cursor:
                    tables = await cursor.fetchall()
                    table_count = len(tables)
                
                # Test write capability
                await db.execute("CREATE TABLE IF NOT EXISTS _test_connection (id INTEGER)")
                await db.execute("DROP TABLE IF EXISTS _test_connection")
                await db.commit()
            
            response_time = (time.time() - start_time) * 1000
            
            return ConnectionResult(
                success=True,
                database_type=DatabaseType.SQLITE,
                connection_info={
                    "path": db_path,
                    "timeout": self.settings.sqlite_timeout
                },
                response_time_ms=response_time,
                server_version=f"SQLite {version}",
                schema_info={
                    "table_count": table_count,
                    "file_size_bytes": os.path.getsize(db_path) if os.path.exists(db_path) else 0
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ConnectionResult(
                success=False,
                database_type=DatabaseType.SQLITE,
                connection_info={"path": self.settings.sqlite_path},
                error_message=str(e),
                response_time_ms=response_time
            )
    
    async def create_connection_url(self, db_type: DatabaseType, **kwargs) -> str:
        """Create a SQLAlchemy connection URL for the specified database type."""
        if db_type == DatabaseType.POSTGRESQL:
            return self.settings.postgres_url
        
        elif db_type == DatabaseType.MYSQL:
            host = kwargs.get('host', self.settings.mysql_host)
            port = kwargs.get('port', self.settings.mysql_port)
            user = kwargs.get('user', self.settings.mysql_username)
            password = kwargs.get('password')
            database = kwargs.get('database', self.settings.mysql_database)
            charset = kwargs.get('charset', self.settings.mysql_charset)
            
            if not password and self.settings.mysql_password:
                password = self.settings.mysql_password.get_secret_value()
            
            if password:
                return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
            else:
                return f"mysql+pymysql://{user}@{host}:{port}/{database}?charset={charset}"
        
        elif db_type == DatabaseType.SQLITE:
            path = kwargs.get('path', self.settings.sqlite_path)
            return f"sqlite+aiosqlite:///{path}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


class DatabaseIntrospector:
    """Introspect database schemas for ingestion."""
    
    def __init__(self, connection_tester: DatabaseConnectionTester):
        self.tester = connection_tester
    
    async def get_schema_info(self, db_type: DatabaseType, **connection_kwargs) -> Dict[str, Any]:
        """Get detailed schema information from a database."""
        connection_url = await self.tester.create_connection_url(db_type, **connection_kwargs)
        
        try:
            if db_type == DatabaseType.SQLITE:
                engine = create_async_engine(connection_url, poolclass=NullPool)
            else:
                engine = create_async_engine(
                    connection_url,
                    pool_size=1,
                    max_overflow=0,
                    pool_timeout=self.tester.settings.connection_timeout
                )
            
            async with engine.begin() as conn:
                if db_type == DatabaseType.POSTGRESQL:
                    return await self._introspect_postgresql(conn)
                elif db_type == DatabaseType.MYSQL:
                    return await self._introspect_mysql(conn)
                elif db_type == DatabaseType.SQLITE:
                    return await self._introspect_sqlite(conn)
            
            await engine.dispose()
            
        except Exception as e:
            logger.error(f"Schema introspection failed for {db_type}: {e}")
            raise
    
    async def _introspect_postgresql(self, conn) -> Dict[str, Any]:
        """Introspect PostgreSQL schema."""
        # Get tables and columns
        result = await conn.execute(sa.text("""
            SELECT 
                t.table_schema,
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale
            FROM information_schema.tables t
            JOIN information_schema.columns c ON t.table_name = c.table_name 
                AND t.table_schema = c.table_schema
            WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY t.table_schema, t.table_name, c.ordinal_position
        """))
        
        tables = {}
        for row in result.fetchall():
            schema_name = row[0]
            table_name = row[1]
            table_key = f"{schema_name}.{table_name}"
            
            if table_key not in tables:
                tables[table_key] = {
                    "schema": schema_name,
                    "name": table_name,
                    "columns": []
                }
            
            tables[table_key]["columns"].append({
                "name": row[2],
                "type": row[3],
                "nullable": row[4] == "YES",
                "default": row[5],
                "max_length": row[6],
                "precision": row[7],
                "scale": row[8]
            })
        
        return {
            "database_type": "postgresql",
            "tables": list(tables.values()),
            "table_count": len(tables)
        }
    
    async def _introspect_mysql(self, conn) -> Dict[str, Any]:
        """Introspect MySQL schema."""
        result = await conn.execute(sa.text("""
            SELECT 
                t.TABLE_SCHEMA,
                t.TABLE_NAME,
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.NUMERIC_PRECISION,
                c.NUMERIC_SCALE
            FROM information_schema.TABLES t
            JOIN information_schema.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME 
                AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
            WHERE t.TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
            ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
        """))
        
        tables = {}
        for row in result.fetchall():
            schema_name = row[0]
            table_name = row[1]
            table_key = f"{schema_name}.{table_name}"
            
            if table_key not in tables:
                tables[table_key] = {
                    "schema": schema_name,
                    "name": table_name,
                    "columns": []
                }
            
            tables[table_key]["columns"].append({
                "name": row[2],
                "type": row[3],
                "nullable": row[4] == "YES",
                "default": row[5],
                "max_length": row[6],
                "precision": row[7],
                "scale": row[8]
            })
        
        return {
            "database_type": "mysql",
            "tables": list(tables.values()),
            "table_count": len(tables)
        }
    
    async def _introspect_sqlite(self, conn) -> Dict[str, Any]:
        """Introspect SQLite schema."""
        # Get tables
        result = await conn.execute(sa.text("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """))
        
        table_names = [row[0] for row in result.fetchall()]
        tables = []
        
        # Get column info for each table
        for table_name in table_names:
            pragma_result = await conn.execute(sa.text(f"PRAGMA table_info({table_name})"))
            columns = []
            
            for col_row in pragma_result.fetchall():
                columns.append({
                    "name": col_row[1],
                    "type": col_row[2],
                    "nullable": not bool(col_row[3]),
                    "default": col_row[4],
                    "primary_key": bool(col_row[5])
                })
            
            tables.append({
                "schema": "main",
                "name": table_name,
                "columns": columns
            })
        
        return {
            "database_type": "sqlite",
            "tables": tables,
            "table_count": len(tables)
        }


