"""
Generic LLM Agent Framework - Claude Only, Synchronous Execution
Simplified version with real Claude API only
"""

import json
import logging
import time
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import shutil

from core.domain_assignment import assign_domains_to_columns, ColumnInfo

logger = logging.getLogger(__name__)

# Global log storage for real-time access
agent_logs: Dict[str, List[Dict[str, Any]]] = {}

@dataclass
class CodeExecution:
    """Result of code execution"""
    code: str
    success: bool
    stdout: str
    stderr: str
    execution_time: float
    files_created: List[str]

@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None

class ClaudeProvider:
    """Anthropic Claude LLM Provider - Synchronous"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-5-haiku-20241022")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.3)
        
        if not self.api_key:
            raise ValueError("Claude API key is required")
        
        # Import anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def generate_code(self, prompt: str, task_type: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate code using Claude"""
        
        system_prompt = self._build_system_prompt(task_type)
        user_prompt = self._build_user_prompt(prompt, task_type, context)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=response.model,
            provider="claude"
        )
    
    def generate_error_correction(self, error_prompt: str, task_type: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate error correction using Claude"""
        
        system_prompt = """You are an expert Python programmer specializing in error correction and debugging.
        Your job is to analyze errors and generate corrected Python code that will work properly.
        Focus on fixing the specific error mentioned while maintaining the intended functionality.
        Return ONLY the corrected Python code, no markdown formatting or explanations."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.1,  # Very low temperature for error correction
            system=system_prompt,
            messages=[{"role": "user", "content": error_prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=response.model,
            provider="claude"
        )
    
    def _build_system_prompt(self, task_type: str) -> str:
        """Build system prompt based on task type"""
        
        base_prompt = """You are an expert Python data scientist and programmer. You generate clean, efficient, and well-documented Python code for data processing tasks.

Your code should:
1. Be production-ready and robust
2. Handle errors gracefully
3. Include informative print statements
4. Use pandas for data manipulation when appropriate
5. Be efficient and follow best practices
6. Include type hints where relevant

You have FULL CONTROL of the entire workflow. Always:
- Detect and handle archive files (e.g., .zip) provided as inputs
- Unzip archives into the current working directory (the workspace), preserving relative paths
- Discover data files after extraction (CSV/TSV/JSON/parquet) and operate on them
- Write all outputs into the current working directory
- Print progress clearly so a user watching a terminal can follow along

Always return ONLY the Python code, no markdown formatting or explanations."""

        task_specific = {
            "exploration": """
Focus on data exploration and analysis:
- Load and examine CSV/TSV files
- Analyze data structure, types, and quality
- Generate summary statistics
- Identify patterns and anomalies
- Create informative output for users""",

            "standardization": """
Focus on data cleaning and standardization:
- Clean column names (lowercase, remove spaces, special chars)
- Standardize data types appropriately
- Handle missing values
- Remove duplicates if necessary
- Create clean, consistent data format""",

            "mapping_preparation": """
Focus on preparing data for domain mapping:
- Extract detailed column metadata
- Sample representative data values
- Identify data patterns and types
- Generate structured metadata for domain assignment
- Create JSON output with column information"""
        }
        
        return f"{base_prompt}\n\n{task_specific.get(task_type, '')}"
    
    def _build_user_prompt(self, prompt: str, task_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build user prompt with context"""
        
        context_str = ""
        if context:
            if "file_analysis" in context:
                fa = context["file_analysis"]
                context_str += f"\nFile Analysis:\n- {len(fa.get('files', []))} files\n- Total size: {fa.get('total_size', 0)} bytes\n- File types: {fa.get('file_types', [])}\n"
                
                for file_info in fa.get('files', [])[:3]:  # Show first 3 files
                    context_str += f"- {file_info.get('name', 'unknown')}: {file_info.get('size', 0)} bytes"
                    if 'sample_header' in file_info:
                        context_str += f" (columns: {file_info.get('sample_header', '')})"
                    context_str += "\n"
            
            if "previous_output" in context and context["previous_output"]:
                context_str += f"\nPrevious step output:\n{context['previous_output'][:500]}...\n"
            
            # Provide explicit instruction to handle archives first
            if "input_files" in context:
                input_list = "\n".join([str(p) for p in context["input_files"]])
                context_str += f"\nInput file paths (some may be archives):\n{input_list}\n"
                context_str += "\nIf any inputs are .zip archives, first unzip them into the current working directory, then proceed.\n"
        
        return f"{prompt}\n{context_str}\n\nGenerate Python code for {task_type}:"

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

class GenericLLMAgent:
    """Generic LLM Agent with Claude provider - Synchronous execution"""
    
    def __init__(self, run_id: str, workspace_dir: Path, provider_config: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Claude provider
        if provider_config is None:
            provider_config = {}
        
        self.llm_provider = ClaudeProvider(provider_config)
        
        # Agent state
        self.last_successful_output = ""
        self.error_history: List[str] = []
        
        log_agent_activity(run_id, "🤖 Claude Agent initialized", "info", "init")
    
    def analyze_and_process_files(self, file_paths: List[Path]) -> Generator[Dict[str, Any], None, None]:
        """Main workflow with Claude provider - Synchronous execution"""
        
        yield {"type": "log", "message": "🤖 Claude Agent starting analysis...", "level": "info"}
        
        # Materialize inputs into workspace for self-contained execution
        try:
            copied_inputs = self._materialize_inputs(file_paths)
            yield {"type": "log", "message": f"📥 Imported {len(copied_inputs)} input file(s) into workspace", "level": "info"}
        except Exception as e:
            yield {"type": "log", "message": f"❌ Failed to import inputs: {e}", "level": "error"}
            copied_inputs = []
        
        # Step 1: Analyze file structure
        yield {"type": "log", "message": "🔍 Analyzing uploaded files...", "level": "info"}
        
        file_analysis = self._analyze_files(copied_inputs or file_paths)
        yield {"type": "analysis", "data": file_analysis}
        
        # Step 2: Data exploration with error recovery
        yield {"type": "log", "message": "💭 Claude thinking about data exploration strategy...", "level": "info"}
        
        for event in self._execute_step_with_retry(
            step_number=1,
            step_name="exploration", 
            reasoning="Exploring data structure and content",
            context={"file_analysis": file_analysis},
            max_retries=2
        ):
            yield event
            if event.get("type") == "step_failed":
                return
        
        # Step 3: Data standardization with error recovery
        yield {"type": "log", "message": "🧠 Claude generating data standardization code...", "level": "info"}
        
        for event in self._execute_step_with_retry(
            step_number=2,
            step_name="standardization",
            reasoning="Standardizing data formats and cleaning", 
            context={"file_analysis": file_analysis, "previous_output": self.last_successful_output},
            max_retries=2
        ):
            yield event
            if event.get("type") == "step_failed":
                return
        
        # Step 4: Domain mapping preparation with error recovery
        yield {"type": "log", "message": "🎯 Claude preparing data for domain mapping...", "level": "info"}
        
        for event in self._execute_step_with_retry(
            step_number=3,
            step_name="mapping_preparation",
            reasoning="Preparing column metadata for domain mapping",
            context={"previous_output": self.last_successful_output},
            max_retries=2
        ):
            yield event
            if event.get("type") == "step_failed":
                return
        
        # Step 5: Deterministic domain mapping assignment (non-LLM)
        yield {"type": "log", "message": "🗂️ Running deterministic domain mapping assignment...", "level": "info"}
        try:
            mapping_result = self._run_domain_mapping_assignment()
            yield {"type": "domain_mapping", "result": mapping_result}
            yield {"type": "log", "message": "✅ Domain mapping assignment completed", "level": "success"}
        except Exception as e:
            yield {"type": "log", "message": f"❌ Domain mapping assignment failed: {e}", "level": "error"}
            # Do not fail the whole run; continue to completion
        
        # Step 6: Organize standardized files to dedicated location
        yield {"type": "log", "message": "📁 Organizing standardized data files...", "level": "info"}
        try:
            self._organize_standardized_files()
            yield {"type": "log", "message": "✅ Standardized files organized", "level": "success"}
        except Exception as e:
            yield {"type": "log", "message": f"⚠️ Failed to organize files: {e}", "level": "warning"}
        
        yield {"type": "log", "message": "🎉 Claude agent completed successfully! Data ready for domain mapping.", "level": "success"}
        yield {"type": "completion", "summary": self._generate_summary()}
    
    def _analyze_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze uploaded files to understand structure"""
        analysis = {
            "files": [],
            "total_size": 0,
            "file_types": set(),
            "estimated_rows": 0
        }
        
        for file_path in file_paths:
            if file_path.exists() and file_path.is_file():
                stat = file_path.stat()
                file_info = {
                    "path": str(file_path),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "extension": file_path.suffix.lower(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                # Quick peek at content for CSV files
                if file_path.suffix.lower() in ['.csv', '.tsv']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline().strip()
                            file_info["sample_header"] = first_line
                            file_info["estimated_columns"] = len(first_line.split(','))
                    except:
                        pass
                
                analysis["files"].append(file_info)
                analysis["total_size"] += stat.st_size
                analysis["file_types"].add(file_path.suffix.lower())
        
        analysis["file_types"] = list(analysis["file_types"])
        # Also provide original input file paths so the LLM can decide to unzip
        analysis["input_files"] = [str(p) for p in file_paths]
        return analysis
    
    def _execute_step_with_retry(self, step_number: int, step_name: str, reasoning: str, 
                                 context: Dict[str, Any], max_retries: int = 2) -> Generator[Dict[str, Any], None, None]:
        """Execute a step with error recovery using Claude - Synchronous"""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt - generate fresh code
                    yield {"type": "log", "message": f"🧠 Claude generating code for {step_name}...", "level": "info"}
                    response = self.llm_provider.generate_code(f"Generate Python code for {step_name}", step_name, context)
                    code = self._extract_code_from_response(response.content)
                else:
                    # Retry attempt - show Claude the error and ask for fix
                    yield {"type": "log", "message": f"🔄 Claude analyzing error and rewriting code (attempt {attempt + 1})...", "level": "warning"}
                    error_context = {
                        **context,
                        "previous_error": self.error_history[-1] if self.error_history else "",
                        "failed_code": getattr(self, 'last_failed_code', ""),
                        "attempt_number": attempt + 1
                    }
                    
                    error_prompt = f"""
PREVIOUS CODE FAILED - PLEASE FIX IT!

Task: {step_name}
Attempt: {attempt + 1}

FAILED CODE:
{getattr(self, 'last_failed_code', '')}

ERROR MESSAGE:
{self.error_history[-1] if self.error_history else ''}

Please analyze the error and write corrected Python code that will work properly.
Focus on fixing the specific error mentioned above.
"""
                    
                    response = self.llm_provider.generate_error_correction(error_prompt, step_name, error_context)
                    code = self._extract_code_from_response(response.content)
                
                yield {"type": "step", "step_number": step_number, "reasoning": reasoning, "code": code, "provider": "claude"}
                
                # Execute the code synchronously
                exec_result = self._execute_code(code, step_name=f"{step_name}_attempt_{attempt + 1}")
                yield {"type": "execution", "result": {
                    "success": exec_result.success,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "execution_time": exec_result.execution_time
                }}
                
                if exec_result.success:
                    # Success! Save output and continue
                    self.last_successful_output = exec_result.stdout
                    yield {"type": "log", "message": f"✅ {step_name.title()} completed successfully", "level": "success"}
                    return
                else:
                    # Execution failed - save error for Claude to see
                    error_msg = f"Code execution failed:\nSTDERR: {exec_result.stderr}\nSTDOUT: {exec_result.stdout}"
                    self.error_history.append(error_msg)
                    self.last_failed_code = code
                    
                    if attempt < max_retries:
                        yield {"type": "log", "message": f"❌ {step_name.title()} failed (attempt {attempt + 1}): {exec_result.stderr[:100]}...", "level": "error"}
                        yield {"type": "log", "message": f"🧠 Claude will analyze the error and try again...", "level": "info"}
                    else:
                        yield {"type": "log", "message": f"❌ {step_name.title()} failed after {max_retries + 1} attempts", "level": "error"}
                        yield {"type": "step_failed", "step": step_name, "final_error": exec_result.stderr}
                        return
                        
            except Exception as e:
                error_msg = f"Unexpected error in {step_name}: {str(e)}"
                self.error_history.append(error_msg)
                
                if attempt < max_retries:
                    yield {"type": "log", "message": f"❌ Unexpected error (attempt {attempt + 1}): {str(e)}", "level": "error"}
                else:
                    yield {"type": "log", "message": f"❌ {step_name.title()} failed with unexpected error after {max_retries + 1} attempts", "level": "error"}
                    yield {"type": "step_failed", "step": step_name, "final_error": str(e)}
                    return
    
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from Claude response, removing markdown formatting"""
        
        # Remove markdown code blocks
        if "```python" in response_content:
            # Extract code between ```python and ```
            start = response_content.find("```python") + 9
            end = response_content.find("```", start)
            if end != -1:
                return response_content[start:end].strip()
        elif "```" in response_content:
            # Extract code between ``` and ```
            start = response_content.find("```") + 3
            end = response_content.find("```", start)
            if end != -1:
                return response_content[start:end].strip()
        
        # If no markdown, return as-is
        return response_content.strip()
    
    def _execute_code(self, code: str, step_name: str) -> CodeExecution:
        """Execute Python code in the workspace - Synchronous execution"""
        start_time = time.time()
        
        # Create code file
        code_file = self.workspace_dir / f"{step_name}_{int(time.time())}.py"
        
        # Add workspace to Python path and imports
        full_code = f"""
import sys
import os
from pathlib import Path
import pandas as pd
import json
import csv
from datetime import datetime

# Set working directory to workspace (absolute path to avoid nesting)
workspace_path = r"{self.workspace_dir.resolve()}"
os.chdir(workspace_path)
sys.path.insert(0, workspace_path)

print(f"Starting {step_name} execution...")
print(f"Working directory: {{os.getcwd()}}")

try:
{self._indent_code(code, 4)}
    print(f"✅ {step_name} completed successfully")
except Exception as e:
    print(f"❌ Error in {step_name}: {{e}}")
    import traceback
    traceback.print_exc()
    raise
"""
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        # Execute the code synchronously
        try:
            # Ensure the code file directory exists
            code_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Set UTF-8 environment for subprocess
            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.workspace_dir.resolve())
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run synchronously
            process = subprocess.run(
                [sys.executable, str(code_file.resolve())],
                capture_output=True,
                text=True,
                cwd=str(self.workspace_dir.resolve()),
                env=env,
                encoding='utf-8',
                errors='ignore'
            )
            
            # Check for created files
            files_before = set(self.workspace_dir.glob('*'))
            files_after = set(self.workspace_dir.glob('*'))
            files_created = [str(f) for f in files_after - files_before]
            
            return CodeExecution(
                code=code,
                success=process.returncode == 0,
                stdout=process.stdout or "",
                stderr=process.stderr or "",
                execution_time=time.time() - start_time,
                files_created=files_created
            )
            
        except Exception as e:
            return CodeExecution(
                code=code,
                success=False,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                files_created=[]
            )
        finally:
            # Cleanup code file
            if code_file.exists():
                try:
                    code_file.unlink()
                except:
                    pass

    def _run_domain_mapping_assignment(self) -> Dict[str, Any]:
        """Collect columns from standardized outputs and run domain assignment.
        Produces domain_assignments.json in the workspace.
        """
        # Prefer domain_mapping_input.json if present
        mapping_input = self.workspace_dir / "domain_mapping_input.json"
        columns: List[ColumnInfo] = []
        source = None
        if mapping_input.exists():
            try:
                data = json.loads(mapping_input.read_text(encoding="utf-8"))
                # Expected structure: { filename: { column_metadata: [ {...} ] } }
                for file_name, payload in data.items():
                    for col in payload.get("column_metadata", []):
                        columns.append(ColumnInfo(
                            name=str(col.get("name", "unknown")),
                            sample_values=[str(v) for v in col.get("sample_values", [])],
                            data_type=str(col.get("data_type", "unknown")),
                            null_count=int(col.get("null_count", 0)),
                            total_count=int(col.get("total_count", 0)),
                            unique_count=int(col.get("unique_count", 0))
                        ))
                source = "domain_mapping_input.json"
            except Exception:
                # Fallback to scan CSVs
                columns = []
        if not columns:
            # Scan standardized CSVs
            csv_candidates = list(self.workspace_dir.glob("standardized_*.csv"))
            if not csv_candidates:
                single = self.workspace_dir / "standardized_data.csv"
                if single.exists():
                    csv_candidates = [single]
            source = "csv_scan"
            for csv_path in csv_candidates:
                try:
                    df = pd.read_csv(csv_path)
                    for col_name in df.columns:
                        series = df[col_name]
                        sample_values = [str(v) for v in series.dropna().astype(str).head(10).tolist()]
                        columns.append(ColumnInfo(
                            name=str(col_name),
                            sample_values=sample_values,
                            data_type=str(series.dtype),
                            null_count=int(series.isna().sum()),
                            total_count=int(len(series)),
                            unique_count=int(series.nunique())
                        ))
                except Exception:
                    continue
        # Run assignment
        assignments = assign_domains_to_columns(columns, run_id=self.run_id)
        # Serialize
        assignments_payload = []
        for a in assignments:
            assignments_payload.append({
                "column_name": a.column_name,
                "domain_id": a.domain_id,
                "domain_name": a.domain_name,
                "confidence_score": a.confidence_score,
                "confidence_band": getattr(a.confidence_band, "value", str(a.confidence_band)),
                "evidence": {
                    "name_similarity": getattr(a.evidence, "name_similarity", 0.0),
                    "regex_strength": getattr(a.evidence, "regex_strength", 0.0),
                    "value_similarity": getattr(a.evidence, "value_similarity", 0.0),
                    "unit_compatibility": getattr(a.evidence, "unit_compatibility", 0.0),
                    "matching_aliases": getattr(a.evidence, "matching_aliases", []),
                    "matching_patterns": getattr(a.evidence, "matching_patterns", []),
                    "matching_units": getattr(a.evidence, "matching_units", []),
                    "header_tokens": getattr(a.evidence, "header_tokens", []),
                    "composite_score": getattr(a.evidence, "composite_score", 0.0),
                }
            })
        out_path = self.workspace_dir / "domain_assignments.json"
        out_path.write_text(json.dumps({
            "run_id": self.run_id,
            "source": source,
            "columns": len(columns),
            "assignments": assignments_payload
        }, indent=2), encoding="utf-8")
        return {
            "output_file": str(out_path),
            "num_columns": len(columns),
            "num_assignments": len(assignments_payload)
        }

    def _materialize_inputs(self, file_paths: List[Path]) -> List[Path]:
        """Copy provided input files into the workspace root so the agent can operate on them locally."""
        copied: List[Path] = []
        for src in file_paths:
            try:
                dest = self.workspace_dir / src.name
                if src.resolve() == dest.resolve():
                    copied.append(dest)
                    continue
                shutil.copy(src, dest)
                copied.append(dest)
            except Exception:
                continue
        return copied
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))
    
    def _organize_standardized_files(self):
        """Move standardized files to dedicated location for domain mapping"""
        try:
            # Create dedicated standardized data directory
            standardized_dir = Path("standardized_data") / self.run_id
            standardized_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all standardized CSV files
            standardized_files = list(self.workspace_dir.glob("**/standardized_*.csv"))
            
            if not standardized_files:
                logger.warning(f"No standardized CSV files found in {self.workspace_dir}")
                return
            
            # Filter out test data files - only organize user data
            user_files = []
            for source_file in standardized_files:
                # Skip files that are clearly test data
                if not any(test_prefix in source_file.name.lower() for test_prefix in ['library_', 'retail_']):
                    user_files.append(source_file)
                else:
                    logger.info(f"🚫 Skipping test data file from organization: {source_file.name}")
            
            if not user_files:
                logger.warning(f"No user data files found to organize - all appear to be test data")
                return
            
            # Copy only user data files to dedicated location
            for source_file in user_files:
                dest_file = standardized_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                logger.info(f"📁 Copied {source_file.name} to standardized data directory")
            
            # Create a metadata file with processing info
            metadata = {
                "run_id": self.run_id,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "standardized_files": [f.name for f in user_files],
                "source_workspace": str(self.workspace_dir),
                "standardized_location": str(standardized_dir),
                "total_files_found": len(standardized_files),
                "user_files_organized": len(user_files),
                "test_files_skipped": len(standardized_files) - len(user_files)
            }
            
            metadata_file = standardized_dir / "processing_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Organized {len(user_files)} user files to {standardized_dir} (skipped {len(standardized_files) - len(user_files)} test files)")
            
        except Exception as e:
            logger.error(f"Failed to organize standardized files: {e}")
            raise
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of agent execution"""
        workspace_files = list(self.workspace_dir.glob('*'))
        
        # Check if standardized files were created
        standardized_dir = Path("standardized_data") / self.run_id
        standardized_files = []
        if standardized_dir.exists():
            standardized_files = [f.name for f in standardized_dir.glob("*.csv")]
        
        return {
            "provider": "claude",
            "model": self.llm_provider.model,
            "total_steps": 3,
            "successful_steps": 3,
            "workspace": str(self.workspace_dir),
            "files_created": [f.name for f in workspace_files if f.is_file()],
            "error_count": len(self.error_history),
            "standardized_files": standardized_files,
            "standardized_location": str(standardized_dir) if standardized_files else None
        }