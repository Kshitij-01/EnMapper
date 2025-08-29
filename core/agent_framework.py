"""
Agent Framework for EnMapper - Code execution and data processing agents
"""

import asyncio
import subprocess
import tempfile
import json
import logging
import time
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import polars as pl
import os
import sys

logger = logging.getLogger(__name__)

# Global log storage for real-time access
agent_logs: Dict[str, List[Dict[str, Any]]] = {}


class ToolType(Enum):
    """Available tool types for agents"""
    PYTHON_EXEC = "python_exec"
    FILE_EXTRACT = "file_extract"
    FILE_READ = "file_read"
    FILE_LIST = "file_list"
    DB_PROFILE = "db_profile"
    DATA_SAMPLE = "data_sample"
    DOMAIN_ASSIGN = "domain_assign"
    PII_SCAN = "pii_scan"


@dataclass
class ToolResult:
    """Result of a tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    artifacts: List[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class AgentContext:
    """Context for agent execution"""
    run_id: str
    workspace_dir: Path
    max_execution_time: int = 120  # seconds
    max_memory_mb: int = 512
    allowed_imports: List[str] = None
    
    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = [
                'pandas', 'polars', 'numpy', 'json', 'csv', 'os', 'pathlib',
                'zipfile', 'datetime', 'typing', 're', 'math', 'statistics'
            ]


class AgentSandbox:
    """Sandboxed execution environment for agents"""
    
    def __init__(self, context: AgentContext):
        self.context = context
        self.workspace_dir = context.workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
    def _validate_python_code(self, code: str) -> Tuple[bool, str]:
        """Validate Python code for security"""
        dangerous_patterns = [
            'import subprocess', 'import os.system', 'exec(', 'eval(',
            '__import__', 'open(', 'file(', 'input(', 'raw_input(',
            'import requests', 'import urllib', 'import socket',
            'rm -', 'del ', 'shutil.rmtree', 'os.remove'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "Code validation passed"
    
    def _create_safe_environment(self) -> Dict[str, str]:
        """Create a safe environment for code execution"""
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.workspace_dir)
        env['PWD'] = str(self.workspace_dir)
        return env
    
    async def execute_python(self, code: str, timeout: Optional[int] = None) -> ToolResult:
        """Execute Python code in a sandboxed environment"""
        start_time = time.time()
        
        # Validate code
        is_safe, validation_msg = self._validate_python_code(code)
        if not is_safe:
            return ToolResult(
                success=False,
                output="",
                error=f"Code validation failed: {validation_msg}",
                execution_time=time.time() - start_time
            )
        
        # Create temporary Python file
        temp_file = self.workspace_dir / f"agent_code_{int(time.time())}.py"
        try:
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # Execute with timeout
            timeout = timeout or self.context.max_execution_time
            env = self._create_safe_environment()
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(temp_file),
                cwd=str(self.workspace_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                success = process.returncode == 0
                output = stdout.decode('utf-8') if stdout else ""
                error = stderr.decode('utf-8') if stderr else None
                
                return ToolResult(
                    success=success,
                    output=output,
                    error=error,
                    execution_time=time.time() - start_time
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Execution timeout after {timeout} seconds",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
    
    def extract_zip(self, zip_path: Path, extract_to: Optional[Path] = None) -> ToolResult:
        """Extract ZIP file safely"""
        start_time = time.time()
        
        if extract_to is None:
            extract_to = self.workspace_dir / "extracted"
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Validate ZIP contents
                for member in zip_ref.namelist():
                    # Prevent directory traversal
                    if '..' in member or member.startswith('/'):
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"Unsafe path in ZIP: {member}",
                            execution_time=time.time() - start_time
                        )
                
                # Extract files
                extracted_files = []
                for member in zip_ref.namelist():
                    if not member.endswith('/'):  # Skip directories
                        extracted_path = extract_to / member
                        extracted_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with zip_ref.open(member) as source, open(extracted_path, 'wb') as target:
                            target.write(source.read())
                        
                        extracted_files.append(str(extracted_path))
                
                return ToolResult(
                    success=True,
                    output=f"Extracted {len(extracted_files)} files",
                    artifacts=extracted_files,
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"ZIP extraction failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def list_files(self, directory: Optional[Path] = None) -> ToolResult:
        """List files in directory"""
        start_time = time.time()
        
        if directory is None:
            directory = self.workspace_dir
        
        try:
            files = []
            for item in directory.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(directory)
                    files.append({
                        'path': str(rel_path),
                        'size': item.stat().st_size,
                        'extension': item.suffix.lower()
                    })
            
            return ToolResult(
                success=True,
                output=json.dumps(files, indent=2),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"File listing failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def read_file_sample(self, file_path: Path, max_rows: int = 100) -> ToolResult:
        """Read a sample from a data file"""
        start_time = time.time()
        
        try:
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}",
                    execution_time=time.time() - start_time
                )
            
            file_ext = file_path.suffix.lower()
            
            # Read based on file type
            if file_ext == '.csv':
                df = pl.read_csv(file_path, n_rows=max_rows, ignore_errors=True)
            elif file_ext == '.tsv':
                df = pl.read_csv(file_path, separator='\t', n_rows=max_rows, ignore_errors=True)
            elif file_ext == '.json':
                df = pl.read_json(file_path)
                if len(df) > max_rows:
                    df = df.head(max_rows)
            elif file_ext == '.parquet':
                df = pl.read_parquet(file_path, n_rows=max_rows)
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)  # First 5KB
                
                return ToolResult(
                    success=True,
                    output=f"Text content preview:\n{content}",
                    execution_time=time.time() - start_time
                )
            
            # Convert to dict for JSON serialization
            sample_data = {
                'shape': df.shape,
                'columns': df.columns,
                'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                'sample': df.to_dicts()[:min(10, len(df))]
            }
            
            return ToolResult(
                success=True,
                output=json.dumps(sample_data, indent=2),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"File reading failed: {str(e)}",
                execution_time=time.time() - start_time
            )


def log_agent_activity(run_id: str, message: str, level: str = "info", step: str = None):
    """Log agent activity for real-time viewing"""
    if run_id not in agent_logs:
        agent_logs[run_id] = []
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "step": step
    }
    
    agent_logs[run_id].append(log_entry)
    logger.info(f"[Agent {run_id[:8]}] {message}")


class DataProcessingAgent:
    """Agent for data processing and domain mapping"""
    
    def __init__(self, context: AgentContext):
        self.context = context
        self.sandbox = AgentSandbox(context)
        log_agent_activity(context.run_id, f"🤖 Agent initialized for run {context.run_id[:8]}", "info", "init")
    
    async def execute_tool(self, tool_type: ToolType, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        logger.info(f"🛠️ Executing tool: {tool_type.value}")
        
        if tool_type == ToolType.PYTHON_EXEC:
            code = kwargs.get('code', '')
            timeout = kwargs.get('timeout')
            return await self.sandbox.execute_python(code, timeout)
        
        elif tool_type == ToolType.FILE_EXTRACT:
            zip_path = Path(kwargs.get('zip_path', ''))
            extract_to = kwargs.get('extract_to')
            if extract_to:
                extract_to = Path(extract_to)
            return self.sandbox.extract_zip(zip_path, extract_to)
        
        elif tool_type == ToolType.FILE_LIST:
            directory = kwargs.get('directory')
            if directory:
                directory = Path(directory)
            return self.sandbox.list_files(directory)
        
        elif tool_type == ToolType.FILE_READ:
            file_path = Path(kwargs.get('file_path', ''))
            max_rows = kwargs.get('max_rows', 100)
            return self.sandbox.read_file_sample(file_path, max_rows)
        
        elif tool_type == ToolType.DATA_SAMPLE:
            return await self._create_data_sample(**kwargs)
        
        elif tool_type == ToolType.DOMAIN_ASSIGN:
            return await self._assign_domains(**kwargs)
        
        elif tool_type == ToolType.PII_SCAN:
            return await self._scan_pii(**kwargs)
        
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool type: {tool_type.value}"
            )
    
    async def _create_data_sample(self, **kwargs) -> ToolResult:
        """Create standardized data sample"""
        start_time = time.time()
        
        try:
            file_path = Path(kwargs.get('file_path', ''))
            
            # Use our existing sampling functionality
            from core.sampling import sampling_policy
            import polars as pl
            
            # Read the file
            df = pl.read_csv(file_path, ignore_errors=True)
            
            # Create sample
            context = {'contains_pii': False, 'privacy_mode': True}
            sample_result = sampling_policy.create_sample_pack(
                df=df, 
                run_id=self.context.run_id, 
                context=context
            )
            
            return ToolResult(
                success=True,
                output=f"Sample created with {len(sample_result.sampled_df)} rows",
                artifacts=[str(self.context.workspace_dir / "sample.csv")],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Sample creation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _assign_domains(self, **kwargs) -> ToolResult:
        """Assign domains to columns"""
        start_time = time.time()
        
        try:
            # Use our existing domain assignment
            from core.domain_assignment import DomainAssignmentEngine, ColumnInfo
            
            columns_data = kwargs.get('columns', [])
            columns = [ColumnInfo(**col) for col in columns_data]
            
            engine = DomainAssignmentEngine()
            assignments = engine.assign_domains(columns, self.context.run_id)
            
            result_data = [
                {
                    'column': a.column_name,
                    'domain': a.domain_name,
                    'confidence': a.confidence_score
                }
                for a in assignments
            ]
            
            return ToolResult(
                success=True,
                output=json.dumps(result_data, indent=2),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Domain assignment failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _scan_pii(self, **kwargs) -> ToolResult:
        """Scan for PII in data"""
        start_time = time.time()
        
        try:
            file_path = Path(kwargs.get('file_path', ''))
            
            # Use our existing PII functionality
            from core.pii import pii_masker
            import polars as pl
            
            df = pl.read_csv(file_path, ignore_errors=True)
            masked_df, metadata = pii_masker.mask_dataframe(df, aggressive=True)
            
            return ToolResult(
                success=True,
                output=json.dumps(metadata, indent=2),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"PII scan failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class AgentOrchestrator:
    """Orchestrates multiple agents for complex data processing workflows"""
    
    def __init__(self):
        self.active_agents: Dict[str, DataProcessingAgent] = {}
    
    def create_agent(self, run_id: str, workspace_dir: Path) -> DataProcessingAgent:
        """Create a new agent for a run"""
        context = AgentContext(run_id=run_id, workspace_dir=workspace_dir)
        agent = DataProcessingAgent(context)
        self.active_agents[run_id] = agent
        return agent
    
    def get_agent(self, run_id: str) -> Optional[DataProcessingAgent]:
        """Get existing agent for a run"""
        return self.active_agents.get(run_id)
    
    def cleanup_agent(self, run_id: str):
        """Cleanup agent resources"""
        if run_id in self.active_agents:
            agent = self.active_agents[run_id]
            # Cleanup workspace if needed
            if agent.context.workspace_dir.exists():
                shutil.rmtree(agent.context.workspace_dir, ignore_errors=True)
            del self.active_agents[run_id]
    
    async def process_file_with_agent(self, run_id: str, file_path: Path) -> Dict[str, Any]:
        """Process a file using agent workflow"""
        logger.info(f"🤖 Starting agent-based file processing for run: {run_id}")
        log_agent_activity(run_id, f"🚀 Starting agent-based file processing", "info", "start")
        
        # Create workspace
        workspace_dir = Path("agent_workspaces") / run_id
        agent = self.create_agent(run_id, workspace_dir)
        log_agent_activity(run_id, f"📁 Created workspace: {workspace_dir}", "info", "workspace")
        
        workflow_results = []
        
        try:
            # Step 1: Extract if ZIP
            if file_path.suffix.lower() == '.zip':
                log_agent_activity(run_id, "📦 Extracting ZIP file...", "info", "extract")
                extract_result = await agent.execute_tool(
                    ToolType.FILE_EXTRACT, 
                    zip_path=str(file_path)
                )
                workflow_results.append({"step": "extract", "result": extract_result})
                
                if not extract_result.success:
                    log_agent_activity(run_id, f"❌ ZIP extraction failed: {extract_result.error}", "error", "extract")
                    raise Exception(f"ZIP extraction failed: {extract_result.error}")
                else:
                    log_agent_activity(run_id, f"✅ Extracted {len(extract_result.artifacts)} files", "success", "extract")
            
            # Step 2: List available files
            log_agent_activity(run_id, "📋 Listing extracted files...", "info", "list_files")
            list_result = await agent.execute_tool(ToolType.FILE_LIST)
            workflow_results.append({"step": "list_files", "result": list_result})
            
            # Step 3: Analyze each data file
            files_data = json.loads(list_result.output) if list_result.success else []
            data_files = [f for f in files_data if f['extension'] in ['.csv', '.tsv', '.json', '.parquet']]
            
            for file_info in data_files[:3]:  # Process up to 3 files
                file_path = workspace_dir / file_info['path']
                
                logger.info(f"🔍 Analyzing file: {file_info['path']}")
                
                # Read sample
                read_result = await agent.execute_tool(
                    ToolType.FILE_READ, 
                    file_path=str(file_path),
                    max_rows=50
                )
                workflow_results.append({
                    "step": f"read_{file_info['path']}", 
                    "result": read_result
                })
                
                if read_result.success:
                    # Parse sample data for domain assignment
                    sample_data = json.loads(read_result.output)
                    columns = []
                    
                    for col_name in sample_data.get('columns', []):
                        # Extract sample values for this column
                        sample_values = [
                            row.get(col_name, '') for row in sample_data.get('sample', [])
                            if row.get(col_name) is not None
                        ][:10]
                        
                        columns.append({
                            'name': col_name,
                            'sample_values': sample_values,
                            'data_type': sample_data.get('dtypes', {}).get(col_name, 'unknown'),
                            'null_count': 0,
                            'total_count': len(sample_values),
                            'unique_count': len(set(sample_values))
                        })
                    
                    # Assign domains
                    if columns:
                        logger.info(f"🎯 Assigning domains for {len(columns)} columns")
                        domain_result = await agent.execute_tool(
                            ToolType.DOMAIN_ASSIGN,
                            columns=columns
                        )
                        workflow_results.append({
                            "step": f"domains_{file_info['path']}", 
                            "result": domain_result
                        })
                    
                    # PII scan
                    logger.info(f"🔒 Scanning for PII")
                    pii_result = await agent.execute_tool(
                        ToolType.PII_SCAN,
                        file_path=str(file_path)
                    )
                    workflow_results.append({
                        "step": f"pii_{file_info['path']}", 
                        "result": pii_result
                    })
            
            # Step 4: Generate Python code for standardization
            logger.info("🐍 Generating standardization code")
            standardization_code = self._generate_standardization_code(workflow_results)
            
            code_result = await agent.execute_tool(
                ToolType.PYTHON_EXEC,
                code=standardization_code,
                timeout=60
            )
            workflow_results.append({"step": "standardization", "result": code_result})
            
            return {
                "success": True,
                "workflow_results": workflow_results,
                "summary": self._generate_workflow_summary(workflow_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Agent workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_results": workflow_results
            }
        
        finally:
            # Cleanup is handled by the orchestrator
            pass
    
    def _generate_standardization_code(self, workflow_results: List[Dict]) -> str:
        """Generate Python code for data standardization"""
        return """
import polars as pl
import json
from pathlib import Path

# Load and standardize data files
standardized_data = {}

print("Starting data standardization...")

try:
    # Find CSV files in workspace
    workspace = Path('.')
    csv_files = list(workspace.glob('**/*.csv'))
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        # Read CSV
        df = pl.read_csv(csv_file, ignore_errors=True)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Basic standardization
        # 1. Clean column names
        df = df.rename({col: col.strip().lower().replace(' ', '_') for col in df.columns})
        
        # 2. Remove completely empty rows
        df = df.filter(pl.any_horizontal(pl.all().is_not_null()))
        
        # 3. Basic data type inference
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                # Try to convert to numeric if possible
                try:
                    df = df.with_columns(pl.col(col).str.strip_chars())
                except:
                    pass
        
        standardized_data[str(csv_file)] = {
            'shape': df.shape,
            'columns': df.columns,
            'sample': df.head(5).to_dicts()
        }
        
        print(f"Standardized {csv_file}: {df.shape}")

    # Output summary
    print(f"\\nStandardization complete!")
    print(f"Processed {len(standardized_data)} files")
    
    for file_path, info in standardized_data.items():
        print(f"  {file_path}: {info['shape'][0]} rows, {info['shape'][1]} columns")

except Exception as e:
    print(f"Error during standardization: {e}")
"""
    
    def _generate_workflow_summary(self, workflow_results: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of the workflow results"""
        summary = {
            'total_steps': len(workflow_results),
            'successful_steps': sum(1 for r in workflow_results if r['result'].success),
            'files_processed': 0,
            'domains_assigned': 0,
            'pii_fields_found': 0
        }
        
        for result in workflow_results:
            step_name = result['step']
            if step_name.startswith('read_'):
                summary['files_processed'] += 1
            elif step_name.startswith('domains_'):
                if result['result'].success:
                    try:
                        domains_data = json.loads(result['result'].output)
                        summary['domains_assigned'] += len(domains_data)
                    except:
                        pass
            elif step_name.startswith('pii_'):
                if result['result'].success:
                    try:
                        pii_data = json.loads(result['result'].output)
                        summary['pii_fields_found'] += len(pii_data.get('pii_fields_detected', []))
                    except:
                        pass
        
        return summary


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()
