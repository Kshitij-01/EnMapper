"""
LLM-Powered Code Generation Agent for Data Processing
"""

import asyncio
import subprocess
import json
import logging
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
class AgentStep:
    """A single step in the agent workflow"""
    step_number: int
    reasoning: str
    code_generated: str
    execution_result: CodeExecution
    timestamp: str

class LLMCodeAgent:
    """LLM-powered agent that generates and executes Python code"""
    
    def __init__(self, run_id: str, workspace_dir: Path):
        self.run_id = run_id
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.steps: List[AgentStep] = []
        self.llm_client = None  # Will initialize based on available providers
        self.last_successful_output = ""  # Track output from successful steps
        self.error_history: List[str] = []  # Track errors for LLM context
        
    async def initialize_llm(self):
        """Initialize LLM client"""
        try:
            # Try to use existing model registry
            from core.providers import ModelRegistry
            registry = ModelRegistry()
            await registry.initialize()
            
            # For now, we'll use a mock LLM that generates reasonable code
            # In production, this would connect to OpenAI, Claude, etc.
            self.llm_client = MockCodeLLM()
            logger.info(f"🤖 LLM agent initialized for run {self.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_client = MockCodeLLM()  # Fallback
    
    async def analyze_and_process_files(self, file_paths: List[Path]) -> AsyncGenerator[Dict[str, Any], None]:
        """Main workflow: analyze files and generate processing code with error recovery"""
        
        if not self.llm_client:
            await self.initialize_llm()
        
        yield {"type": "log", "message": "🤖 LLM Agent starting analysis...", "level": "info"}
        
        # Step 1: Analyze file structure
        yield {"type": "log", "message": "🔍 Analyzing uploaded files...", "level": "info"}
        
        file_analysis = await self._analyze_files(file_paths)
        yield {"type": "analysis", "data": file_analysis}
        
        # Step 2: Data exploration with error recovery
        yield {"type": "log", "message": "💭 LLM thinking about data exploration strategy...", "level": "info"}
        
        async for event in self._execute_step_with_retry(
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
        yield {"type": "log", "message": "🧠 LLM generating data standardization code...", "level": "info"}
        
        async for event in self._execute_step_with_retry(
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
        yield {"type": "log", "message": "🎯 LLM preparing data for domain mapping...", "level": "info"}
        
        async for event in self._execute_step_with_retry(
            step_number=3,
            step_name="mapping_preparation",
            reasoning="Preparing column metadata for domain mapping",
            context={"previous_output": self.last_successful_output},
            max_retries=2
        ):
            yield event
            if event.get("type") == "step_failed":
                return
        
        yield {"type": "log", "message": "🎉 LLM agent completed successfully! Data ready for domain mapping.", "level": "success"}
        yield {"type": "completion", "summary": self._generate_summary()}
    
    async def _analyze_files(self, file_paths: List[Path]) -> Dict[str, Any]:
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
        return analysis
    
    async def _execute_step_with_retry(self, step_number: int, step_name: str, reasoning: str, 
                                     context: Dict[str, Any], max_retries: int = 2) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a step with error recovery - let LLM see errors and rewrite code"""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt - generate fresh code
                    yield {"type": "log", "message": f"🧠 LLM generating code for {step_name}...", "level": "info"}
                    code = await self._generate_code_for_step(step_name, context)
                else:
                    # Retry attempt - show LLM the error and ask for fix
                    yield {"type": "log", "message": f"🔄 LLM analyzing error and rewriting code (attempt {attempt + 1})...", "level": "warning"}
                    error_context = {
                        **context,
                        "previous_error": self.error_history[-1] if self.error_history else "",
                        "failed_code": self.last_failed_code if hasattr(self, 'last_failed_code') else "",
                        "attempt_number": attempt + 1
                    }
                    code = await self._generate_error_corrected_code(step_name, error_context)
                
                yield {"type": "step", "step_number": step_number, "reasoning": reasoning, "code": code}
                
                # Execute the code
                exec_result = await self._execute_code(code, step_name=f"{step_name}_attempt_{attempt + 1}")
                yield {"type": "execution", "result": exec_result}
                
                if exec_result.success:
                    # Success! Save output and continue
                    self.last_successful_output = exec_result.stdout
                    yield {"type": "log", "message": f"✅ {step_name.title()} completed successfully", "level": "success"}
                    return
                else:
                    # Execution failed - save error for LLM to see
                    error_msg = f"Code execution failed:\nSTDERR: {exec_result.stderr}\nSTDOUT: {exec_result.stdout}"
                    self.error_history.append(error_msg)
                    self.last_failed_code = code
                    
                    if attempt < max_retries:
                        yield {"type": "log", "message": f"❌ {step_name.title()} failed (attempt {attempt + 1}): {exec_result.stderr[:100]}...", "level": "error"}
                        yield {"type": "log", "message": f"🧠 LLM will analyze the error and try again...", "level": "info"}
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
    
    async def _generate_code_for_step(self, step_name: str, context: Dict[str, Any]) -> str:
        """Generate code for a specific step"""
        if step_name == "exploration":
            return await self._generate_exploration_code(context.get("file_analysis", {}))
        elif step_name == "standardization":
            return await self._generate_standardization_code(context.get("file_analysis", {}), context.get("previous_output", ""))
        elif step_name == "mapping_preparation":
            return await self._generate_mapping_prep_code(context.get("previous_output", ""))
        else:
            return await self.llm_client.generate_code(f"Generate code for {step_name}", step_name)
    
    async def _generate_error_corrected_code(self, step_name: str, error_context: Dict[str, Any]) -> str:
        """Let LLM see the error and generate corrected code"""
        
        # Build error analysis prompt for the LLM
        error_prompt = f"""
PREVIOUS CODE FAILED - PLEASE FIX IT!

Task: {step_name}
Attempt: {error_context.get('attempt_number', 1)}

FAILED CODE:
{error_context.get('failed_code', '')}

ERROR MESSAGE:
{error_context.get('previous_error', '')}

CONTEXT:
{error_context.get('file_analysis', {})}
Previous successful output: {error_context.get('previous_output', '')}

Please analyze the error and write corrected Python code that will work properly.
Focus on fixing the specific error mentioned above.
"""
        
        # Use the LLM to generate corrected code
        corrected_code = await self.llm_client.generate_error_correction(error_prompt, step_name)
        return corrected_code
    
    async def _generate_exploration_code(self, file_analysis: Dict[str, Any]) -> str:
        """Generate code to explore the data files"""
        
        # Simulate LLM reasoning about the files
        prompt = f"""
        I need to analyze these files: {[f['name'] for f in file_analysis['files']]}
        File types: {file_analysis['file_types']}
        Total size: {file_analysis['total_size']} bytes
        
        Generate Python code to explore and understand the data structure.
        """
        
        code = await self.llm_client.generate_code(prompt, "exploration")
        return code
    
    async def _generate_standardization_code(self, file_analysis: Dict[str, Any], exploration_output: str) -> str:
        """Generate code to standardize the data"""
        
        prompt = f"""
        Based on exploration results: {exploration_output[:500]}...
        Files to process: {[f['name'] for f in file_analysis['files']]}
        
        Generate Python code to standardize and clean the data for further processing.
        """
        
        code = await self.llm_client.generate_code(prompt, "standardization")
        return code
    
    async def _generate_mapping_prep_code(self, standardization_output: str) -> str:
        """Generate code to prepare data for domain mapping"""
        
        prompt = f"""
        After standardization: {standardization_output[:500]}...
        
        Generate Python code to prepare column metadata and samples for domain mapping.
        """
        
        code = await self.llm_client.generate_code(prompt, "mapping_preparation")
        return code
    
    async def _execute_code(self, code: str, step_name: str) -> CodeExecution:
        """Execute Python code in the workspace with real-time output"""
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
        
        # Execute the code
        try:
            # Ensure the code file directory exists
            code_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Set UTF-8 environment for subprocess
            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.workspace_dir.resolve())
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(code_file.resolve()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir.resolve()),
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            # Check for created files
            files_before = set(self.workspace_dir.glob('*'))
            files_after = set(self.workspace_dir.glob('*'))
            files_created = [str(f) for f in files_after - files_before]
            
            return CodeExecution(
                code=code,
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore'),
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
                code_file.unlink()
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of agent execution"""
        return {
            "total_steps": len(self.steps),
            "successful_steps": sum(1 for step in self.steps if step.execution_result.success),
            "total_execution_time": sum(step.execution_result.execution_time for step in self.steps),
            "files_created": sum(step.execution_result.files_created for step in self.steps),
            "workspace": str(self.workspace_dir)
        }


class MockCodeLLM:
    """Mock LLM that generates reasonable Python code for data processing"""
    
    async def generate_code(self, prompt: str, task_type: str) -> str:
        """Generate appropriate code based on task type"""
        
        if task_type == "exploration":
            return '''
# Data Exploration Code
import pandas as pd
import os
from pathlib import Path

print("🔍 Exploring uploaded files...")

workspace = Path(".")
data_files = list(workspace.glob("*.csv")) + list(workspace.glob("*.tsv")) + list(workspace.glob("*.json"))

print(f"Found {len(data_files)} data files")

for file_path in data_files:
    print(f"\\n📄 Analyzing: {file_path.name}")
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows
        elif file_path.suffix.lower() == '.tsv':
            df = pd.read_csv(file_path, sep='\\t', nrows=5)
        else:
            continue
            
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types: {dict(df.dtypes)}")
        print(f"  Sample data:")
        print(df.head(2).to_string(index=False))
        
    except Exception as e:
        print(f"  ❌ Error reading {file_path.name}: {e}")

print("\\n🎯 Exploration complete!")
'''
        
        elif task_type == "standardization":
            return '''
# Data Standardization Code
import pandas as pd
import re
from pathlib import Path

print("🔧 Starting data standardization...")

def clean_column_names(df):
    """Clean and standardize column names"""
    df.columns = df.columns.str.strip()  # Remove whitespace
    df.columns = df.columns.str.lower()  # Lowercase
    df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)  # Remove special chars
    return df

def standardize_data_types(df):
    """Standardize common data types"""
    for col in df.columns:
        # Try to convert to numeric if possible
        if df[col].dtype == 'object':
            # Check if it looks like a number
            numeric_count = df[col].str.match(r'^-?\\d+\\.?\\d*$').sum()
            if numeric_count > len(df) * 0.8:  # 80% numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  ✅ Converted {col} to numeric")
                except:
                    pass
    return df

# Process each CSV file
workspace = Path(".")
csv_files = list(workspace.glob("*.csv"))

standardized_files = []

for file_path in csv_files:
    print(f"\\n📊 Standardizing: {file_path.name}")
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        original_shape = df.shape
        
        # Clean column names
        df = clean_column_names(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Standardize data types
        df = standardize_data_types(df)
        
        # Save standardized version
        output_file = workspace / f"standardized_{file_path.name}"
        df.to_csv(output_file, index=False)
        
        print(f"  ✅ Processed {original_shape} → {df.shape}")
        print(f"  💾 Saved: {output_file.name}")
        
        standardized_files.append({
            "original": file_path.name,
            "standardized": output_file.name,
            "columns": list(df.columns),
            "shape": df.shape
        })
        
    except Exception as e:
        print(f"  ❌ Error standardizing {file_path.name}: {e}")

print(f"\\n🎉 Standardized {len(standardized_files)} files!")

# Save metadata
import json
with open("standardization_metadata.json", "w") as f:
    json.dump(standardized_files, f, indent=2)

print("💾 Saved standardization metadata")
'''
        
        elif task_type == "mapping_preparation":
            return '''
# Mapping Preparation Code
import pandas as pd
import json
from pathlib import Path

print("🎯 Preparing data for domain mapping...")

def extract_column_metadata(df, filename):
    """Extract metadata for each column for domain mapping"""
    metadata = []
    
    for col in df.columns:
        col_data = {
            "name": col,
            "data_type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "total_count": len(df),
            "unique_count": int(df[col].nunique()),
            "sample_values": []
        }
        
        # Get sample values (non-null)
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_size = min(10, len(non_null_values))
            col_data["sample_values"] = [str(val) for val in non_null_values.head(sample_size).tolist()]
        
        metadata.append(col_data)
    
    return metadata

# Process standardized files
workspace = Path(".")
standardized_files = list(workspace.glob("standardized_*.csv"))

mapping_data = {}

for file_path in standardized_files:
    print(f"\\n🔍 Processing: {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Extract metadata for domain mapping
        column_metadata = extract_column_metadata(df, file_path.name)
        
        mapping_data[file_path.name] = {
            "file_info": {
                "name": file_path.name,
                "shape": df.shape,
                "columns": list(df.columns)
            },
            "column_metadata": column_metadata
        }
        
        print(f"  ✅ Extracted metadata for {len(column_metadata)} columns")
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")

# Save mapping preparation data
with open("domain_mapping_input.json", "w") as f:
    json.dump(mapping_data, f, indent=2)

print(f"\\n🎉 Prepared mapping data for {len(mapping_data)} files!")
print("💾 Saved domain_mapping_input.json")

# Summary
total_columns = sum(len(data["column_metadata"]) for data in mapping_data.values())
print(f"📊 Total columns ready for domain mapping: {total_columns}")
'''
        
        else:
            return '''
print("Hello from LLM-generated code!")
import os
print(f"Working directory: {os.getcwd()}")
'''
    
    async def generate_error_correction(self, error_prompt: str, task_type: str) -> str:
        """Generate corrected code based on error analysis"""
        
        # Analyze the error and generate fixed code
        if "pandas" in error_prompt.lower() and "modulenotfounderror" in error_prompt.lower():
            # Fix missing pandas import
            return self._generate_fixed_pandas_code(task_type)
        elif "filenotfound" in error_prompt.lower() or "no such file" in error_prompt.lower():
            # Fix file path issues
            return self._generate_fixed_filepath_code(task_type)
        elif "syntaxerror" in error_prompt.lower():
            # Fix syntax errors
            return self._generate_fixed_syntax_code(task_type)
        else:
            # Generic error fix - add more error handling
            return self._generate_robust_code(task_type)
    
    def _generate_fixed_pandas_code(self, task_type: str) -> str:
        """Generate code with proper pandas installation check"""
        base_code = '''
# First, ensure pandas is available
try:
    import pandas as pd
    print("✅ Pandas is available")
except ImportError:
    print("❌ Pandas not found, installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
    print("✅ Pandas installed and imported")

import os
from pathlib import Path

print("🔍 Starting data exploration with error recovery...")
'''
        
        if task_type == "exploration":
            return base_code + '''
workspace = Path(".")
print(f"📁 Working in: {workspace.resolve()}")

# Find data files safely
data_files = []
for pattern in ["*.csv", "*.tsv", "*.json"]:
    data_files.extend(list(workspace.glob(pattern)))

print(f"Found {len(data_files)} data files")

for file_path in data_files:
    print(f"\\n📄 Analyzing: {file_path.name}")
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=5, encoding='utf-8', errors='ignore')
        elif file_path.suffix.lower() == '.tsv':
            df = pd.read_csv(file_path, sep='\\t', nrows=5, encoding='utf-8', errors='ignore')
        else:
            print(f"  ⏩ Skipping {file_path.suffix} files for now")
            continue
            
        print(f"  📊 Shape: {df.shape}")
        print(f"  📋 Columns: {list(df.columns)}")
        print(f"  🔧 Data types: {dict(df.dtypes)}")
        print(f"  📝 Sample data:")
        print(df.head(2).to_string(index=False))
        
    except Exception as e:
        print(f"  ❌ Error reading {file_path.name}: {e}")
        print(f"  🔄 Trying with different encoding...")
        try:
            df = pd.read_csv(file_path, nrows=5, encoding='latin-1', errors='ignore')
            print(f"  ✅ Success with latin-1 encoding!")
            print(f"  📊 Shape: {df.shape}")
        except Exception as e2:
            print(f"  ❌ Still failed: {e2}")

print("\\n🎯 Exploration complete with error recovery!")
'''
        else:
            return base_code + '''
print("🔧 Error-corrected code execution")
'''
    
    def _generate_fixed_filepath_code(self, task_type: str) -> str:
        """Generate code with robust file path handling"""
        return '''
import os
from pathlib import Path
import pandas as pd

print("🔍 Starting with robust file handling...")

# Get absolute workspace path
workspace = Path(".").resolve()
print(f"📁 Absolute workspace: {workspace}")

# List all files first
all_files = list(workspace.glob("*"))
print(f"📋 All files in workspace: {[f.name for f in all_files if f.is_file()]}")

# Find CSV files with multiple approaches
csv_files = []
for pattern in ["*.csv", "*.CSV", "**/*.csv"]:
    found_files = list(workspace.glob(pattern))
    csv_files.extend(found_files)

# Remove duplicates
csv_files = list(set(csv_files))
print(f"📊 Found {len(csv_files)} CSV files")

for csv_file in csv_files:
    if csv_file.exists() and csv_file.is_file():
        print(f"\\n📄 Processing: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file, nrows=10)
            print(f"  ✅ Successfully read {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    else:
        print(f"❌ File not found or not accessible: {csv_file}")

print("\\n🎯 File processing complete!")
'''
    
    def _generate_fixed_syntax_code(self, task_type: str) -> str:
        """Generate code with fixed syntax"""
        return '''
import pandas as pd
import os
from pathlib import Path

print("🔧 Running syntax-corrected code...")

try:
    workspace = Path(".")
    csv_files = list(workspace.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        df = pd.read_csv(csv_file, nrows=5)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    print("✅ Syntax-corrected execution completed!")
    
except Exception as e:
    print(f"❌ Still encountering error: {e}")
    print("🔄 Will try alternative approach...")
'''
    
    def _generate_robust_code(self, task_type: str) -> str:
        """Generate robust code with extensive error handling"""
        return '''
print("🛡️ Running robust error-resistant code...")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")
    import sys
    print("🔄 Attempting to install pandas...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
    import pandas as pd
    print("✅ Pandas installed and imported")

import os
from pathlib import Path
import traceback

try:
    workspace = Path(".").resolve()
    print(f"📁 Working directory: {workspace}")
    
    # List everything in workspace
    all_items = list(workspace.iterdir())
    files = [item for item in all_items if item.is_file()]
    
    print(f"📋 Files found: {[f.name for f in files]}")
    
    # Process CSV files with extensive error handling
    csv_files = [f for f in files if f.suffix.lower() == '.csv']
    print(f"📊 CSV files: {len(csv_files)}")
    
    for csv_file in csv_files:
        try:
            print(f"\\n📄 Processing: {csv_file.name}")
            
            # Try multiple encoding options
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, nrows=5, encoding=encoding)
                    print(f"  ✅ Success with {encoding} encoding")
                    print(f"  📊 Shape: {df.shape}")
                    print(f"  📋 Columns: {list(df.columns)}")
                    break
                except Exception as e:
                    print(f"  ⚠️ Failed with {encoding}: {e}")
                    continue
            else:
                print(f"  ❌ Failed with all encodings")
                
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            traceback.print_exc()
    
    print("\\n🎯 Robust processing completed!")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    traceback.print_exc()
'''
