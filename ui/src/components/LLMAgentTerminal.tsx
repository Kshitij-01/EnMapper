import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Chip,
  Stack,
  Fade,
  Collapse,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import {
  SmartToy as AgentIcon,
  Code as CodeIcon,
  Terminal as TerminalIcon,
  Close as CloseIcon,
  PlayArrow as PlayIcon,
} from '@mui/icons-material';

interface LLMStep {
  step_number: number;
  reasoning: string;
  code: string;
  execution_result?: {
    success: boolean;
    stdout: string;
    stderr: string;
    execution_time: number;
  };
  id?: string;
  timestamp?: string;
}

interface LLMAgentTerminalProps {
  runId: string;
  isVisible: boolean;
  onClose: () => void;
  onComplete?: () => void;
}

const LLMAgentTerminal: React.FC<LLMAgentTerminalProps> = ({ runId, isVisible, onClose, onComplete }) => {
  const [logs, setLogs] = useState<any[]>([]);
  const [steps, setSteps] = useState<LLMStep[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [analysis, setAnalysis] = useState<any>(null);
  const [codeExplanationOpen, setCodeExplanationOpen] = useState(false);
  const [selectedCode, setSelectedCode] = useState<LLMStep | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const completedRef = useRef<boolean>(false);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs, steps]);

  // Auto-start processing when terminal becomes visible or runId changes
  useEffect(() => {
    if (isVisible && runId && !isProcessing && !completedRef.current) {
      console.log('🚀 LLM Agent Terminal starting processing for runId:', runId);
      startLLMProcessing();
    } else if (isVisible && !runId) {
      console.log('⚠️ LLM Agent Terminal is visible but runId is null/undefined');
    }
    return () => {
      // Cleanup on unmount or hide
      if (!isVisible && eventSourceRef.current) {
        console.log('🧹 Cleaning up EventSource connection');
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, runId]);

  const startLLMProcessing = async () => {
    if (!runId) {
      console.log('❌ Cannot start LLM processing: runId is null/undefined');
      return;
    }
    if (completedRef.current) {
      console.log('⏭️ Processing already completed for this run; not restarting.');
      return;
    }
    
    console.log('🚀 Starting LLM processing for runId:', runId);
    setIsProcessing(true);
    completedRef.current = false;
    setLogs([]);
    setSteps([]);
    setActiveStep(0);
    
    try {
      // Add initial log
      setLogs([{
        type: 'log',
        message: `🔌 Connecting to LLM Agent stream for run: ${runId.slice(0, 8)}...`,
        level: 'info',
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      // Start LLM processing with EventSource (GET variant) for real-time updates
      // Close any existing connection
      if (eventSourceRef.current) {
        try {
          eventSourceRef.current.close();
        } catch (e) {
          console.warn('Error closing EventSource:', e);
        }
        eventSourceRef.current = null;
      }
      
      const streamUrl = `http://localhost:8000/api/v1/agent/llm-stream/${runId}`;
      console.log('🌐 Connecting to EventSource:', streamUrl);
      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;
      
      eventSource.onopen = () => {
        setLogs(prev => [...prev, {
          type: 'log',
          message: '✅ Connected to LLM Agent stream',
          level: 'success',
          timestamp: new Date().toLocaleTimeString()
        }]);
      };
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleLLMEvent(data);
        } catch (error) {
          console.error('Failed to parse LLM event:', error);
          setLogs(prev => [...prev, {
            type: 'log',
            message: `❌ Failed to parse event: ${error}`,
            level: 'error',
            timestamp: new Date().toLocaleTimeString()
          }]);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error, 'ReadyState:', eventSource.readyState);
        
        let errorMessage = '❌ Connection to LLM Agent failed';
        if (eventSource.readyState === EventSource.CONNECTING) {
          errorMessage = '❌ Failed to connect to LLM Agent stream - retrying...';
        } else if (eventSource.readyState === EventSource.CLOSED) {
          errorMessage = '❌ Connection to LLM Agent lost';
        }
        
        setLogs(prev => [...prev, {
          type: 'log',
          message: errorMessage,
          level: 'error',
          timestamp: new Date().toLocaleTimeString()
        }]);
        
        // Only stop processing if the connection is definitively closed
        if (eventSource.readyState === EventSource.CLOSED) {
          setIsProcessing(false);
          eventSource.close();
          eventSourceRef.current = null;
        }
      };
      
    } catch (error) {
      console.error('LLM processing failed:', error);
      setLogs(prev => [...prev, {
        type: 'log',
        message: `❌ LLM processing failed: ${error}`,
        level: 'error',
        timestamp: new Date().toLocaleTimeString()
      }]);
      setIsProcessing(false);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const simulateLLMProcessing = async (filePaths: string[]) => {
    // Simulate LLM agent workflow
    const events = [
      { type: 'log', message: '🤖 LLM Agent starting analysis...', level: 'info' },
      { type: 'log', message: '🔍 Analyzing uploaded files...', level: 'info' },
      { 
        type: 'analysis', 
        data: { 
          files: filePaths.map(p => ({ name: p.split('/').pop(), size: 5000 })),
          total_size: 15000,
          file_types: ['.csv']
        }
      },
      { type: 'log', message: '💭 LLM thinking about data exploration strategy...', level: 'info' },
      {
        type: 'step',
        step_number: 1,
        reasoning: 'Exploring data structure and content',
        code: `# Data Exploration Code
import pandas as pd
import os
from pathlib import Path

print("🔍 Exploring uploaded files...")

workspace = Path(".")
data_files = list(workspace.glob("*.csv"))

for file_path in data_files:
    print(f"📄 Analyzing: {file_path.name}")
    
    try:
        df = pd.read_csv(file_path, nrows=5)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types: {dict(df.dtypes)}")
        print(df.head(2).to_string(index=False))
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("🎯 Exploration complete!")`
      },
      {
        type: 'execution',
        result: {
          success: true,
          stdout: `🔍 Exploring uploaded files...
📄 Analyzing: netsuite_items_samples_100.csv
  Shape: (100, 23)
  Columns: ['itemid', 'displayname', 'salesdescription', 'baseprice', 'cost']
  Data types: {'itemid': 'object', 'displayname': 'object', 'baseprice': 'float64'}
📄 Analyzing: shopify_product_sample_data_100.csv
  Shape: (100, 32)
  Columns: ['Handle', 'Title', 'Body (HTML)', 'Vendor', 'Type']
🎯 Exploration complete!`,
          stderr: '',
          execution_time: 0.45
        }
      },
      { type: 'log', message: '🧠 LLM generating data standardization code...', level: 'info' },
      {
        type: 'step',
        step_number: 2,
        reasoning: 'Standardizing data formats and cleaning',
        code: `# Data Standardization Code
import pandas as pd
import re

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    return df

def standardize_data_types(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_count = df[col].str.match(r'^-?\\d+\\.?\\d*$').sum()
            if numeric_count > len(df) * 0.8:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Process files
for csv_file in workspace.glob("*.csv"):
    df = pd.read_csv(csv_file)
    df = clean_column_names(df)
    df = standardize_data_types(df)
    
    output_file = f"standardized_{csv_file.name}"
    df.to_csv(output_file, index=False)
    print(f"✅ Standardized: {csv_file.name} → {output_file}")`
      },
      {
        type: 'execution',
        result: {
          success: true,
          stdout: `✅ Standardized: netsuite_items_samples_100.csv → standardized_netsuite_items_samples_100.csv
✅ Standardized: shopify_product_sample_data_100.csv → standardized_shopify_product_sample_data_100.csv`,
          stderr: '',
          execution_time: 0.82
        }
      },
      { type: 'log', message: '🎯 LLM preparing data for domain mapping...', level: 'info' },
      {
        type: 'step',
        step_number: 3,
        reasoning: 'Preparing column metadata for domain mapping',
        code: `# Domain Mapping Preparation
import json

def extract_column_metadata(df, filename):
    metadata = []
    for col in df.columns:
        col_data = {
            "name": col,
            "data_type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "total_count": len(df),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(10).astype(str).tolist()
        }
        metadata.append(col_data)
    return metadata

mapping_data = {}
for csv_file in workspace.glob("standardized_*.csv"):
    df = pd.read_csv(csv_file)
    metadata = extract_column_metadata(df, csv_file.name)
    mapping_data[csv_file.name] = {
        "file_info": {"name": csv_file.name, "shape": df.shape},
        "column_metadata": metadata
    }

with open("domain_mapping_input.json", "w") as f:
    json.dump(mapping_data, f, indent=2)

total_columns = sum(len(data["column_metadata"]) for data in mapping_data.values())
print(f"🎉 Prepared {len(mapping_data)} files with {total_columns} columns for domain mapping!")`
      },
      {
        type: 'execution',
        result: {
          success: true,
          stdout: `🎉 Prepared 2 files with 55 columns for domain mapping!`,
          stderr: '',
          execution_time: 0.23
        }
      },
      { type: 'log', message: '🎉 LLM agent completed successfully! Data ready for domain mapping.', level: 'success' },
      { type: 'completion', summary: { total_steps: 3, successful_steps: 3, files_processed: 2 } }
    ];

    // Simulate real-time processing
    for (let i = 0; i < events.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
      handleLLMEvent(events[i]);
    }
    
    setIsProcessing(false);
  };

  const handleLLMEvent = (event: any) => {
    const { type } = event;
    
    if (type === 'log') {
      setLogs(prev => [...prev, {
        ...event,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } else if (type === 'analysis') {
      setAnalysis(event.data);
    } else if (type === 'step') {
      setSteps(prev => [...prev, { ...event, id: `step-${event.step_number}-${Date.now()}-${Math.random()}` }]);
      setActiveStep(event.step_number - 1);
    } else if (type === 'execution') {
      setSteps(prev => prev.map((step, index) => 
        index === activeStep ? { ...step, execution_result: event.result } : step
      ));
      if (event.result.success) {
        setActiveStep(prev => prev + 1);
      }
    } else if (type === 'completion' || type === 'complete') {
      setLogs(prev => [...prev, {
        type: 'log',
        message: '🎉 All processing completed successfully!',
        level: 'success',
        timestamp: new Date().toLocaleTimeString()
      }]);
      // Stop streaming and prevent EventSource auto-reconnect
      try {
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
      } catch {}
      completedRef.current = true;
      // Notify parent that LLM processing is complete
      setTimeout(() => {
        if (onComplete) {
          onComplete();
        }
      }, 2000); // Wait 2 seconds before triggering next step
      setIsProcessing(false);
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'error': return '#f44336';
      default: return '#2196f3';
    }
  };

  return (
    <Collapse in={isVisible}>
      <Paper
        elevation={4}
        sx={{
          mt: 2,
          backgroundColor: '#1e1e1e',
          color: '#ffffff',
          border: '1px solid #333',
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 2,
            backgroundColor: '#2d2d2d',
            borderBottom: '1px solid #444'
          }}
        >
          <Stack direction="row" spacing={2} alignItems="center">
            <AgentIcon color="primary" />
            <Typography variant="h6" sx={{ color: '#00ff00' }}>
              🧠 LLM Code Agent - Run {runId.slice(0, 8)}
            </Typography>
            {isProcessing && (
              <Chip
                icon={<PlayIcon />}
                label="Generating Code"
                size="small"
                color="warning"
                variant="outlined"
              />
            )}
          </Stack>
          
          <Stack direction="row" spacing={1}>
            {!isProcessing && (
              <IconButton 
                onClick={startLLMProcessing} 
                size="small" 
                sx={{ color: '#4caf50' }}
                title="Start LLM Processing"
              >
                <PlayIcon />
              </IconButton>
            )}
            <IconButton onClick={onClose} size="small" sx={{ color: '#fff' }}>
              <CloseIcon />
            </IconButton>
          </Stack>
        </Box>

        {/* Analysis Results */}
        {analysis && (
          <Box sx={{ p: 2, backgroundColor: '#252525' }}>
            <Typography variant="body2" sx={{ color: '#888', mb: 1 }}>
              📊 File Analysis:
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip label={`${analysis.files?.length || 0} files`} size="small" color="info" />
              <Chip label={`${analysis.total_size || 0} bytes`} size="small" color="info" />
              <Chip label={analysis.file_types?.join(', ') || ''} size="small" color="info" />
            </Stack>
          </Box>
        )}

        {/* LLM Steps */}
        {steps.length > 0 && (
          <Box sx={{ p: 2 }}>
            <Stepper activeStep={activeStep} orientation="vertical">
              {steps.map((step, index) => (
                <Step key={step.id || `step-${step.step_number}-${index}-${step.timestamp || Date.now()}`} expanded={true}>
                  <StepLabel
                    sx={{
                      '& .MuiStepLabel-label': { color: '#fff', fontWeight: 600 }
                    }}
                  >
                    🧠 {step.reasoning}
                  </StepLabel>
                  <StepContent>
                    <Box sx={{ mb: 2 }}>
                      {/* Generated Code */}
                      <Paper 
                        sx={{ 
                          p: 2, 
                          backgroundColor: '#0d1117', 
                          mb: 2,
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: '#161b22',
                          }
                        }}
                        onClick={() => {
                          setSelectedCode(step);
                          setCodeExplanationOpen(true);
                        }}
                      >
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                          <CodeIcon sx={{ color: '#58a6ff' }} />
                          <Typography variant="subtitle2" sx={{ color: '#58a6ff' }}>
                            LLM Generated Code (Click to explain):
                          </Typography>
                        </Stack>
                        <Typography
                          component="pre"
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                            color: '#e6edf3',
                            whiteSpace: 'pre-wrap',
                            maxHeight: 300,
                            overflowY: 'auto'
                          }}
                        >
                          {step.code}
                        </Typography>
                      </Paper>

                      {/* Execution Result */}
                      {step.execution_result && (
                        <Paper sx={{ p: 2, backgroundColor: step.execution_result.success ? '#0f1419' : '#2d1b1b' }}>
                          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                            <TerminalIcon sx={{ color: step.execution_result.success ? '#56d364' : '#f85149' }} />
                            <Typography variant="subtitle2" sx={{ color: step.execution_result.success ? '#56d364' : '#f85149' }}>
                              {step.execution_result.success ? '✅ Execution Output:' : '❌ Execution Error:'}
                            </Typography>
                            <Chip 
                              label={`${step.execution_result.execution_time.toFixed(2)}s`}
                              size="small"
                              variant="outlined"
                            />
                          </Stack>
                          <Typography
                            component="pre"
                            sx={{
                              fontFamily: 'monospace',
                              fontSize: '0.75rem',
                              color: step.execution_result.success ? '#e6edf3' : '#f85149',
                              whiteSpace: 'pre-wrap',
                              maxHeight: 150,
                              overflowY: 'auto'
                            }}
                          >
                            {step.execution_result.success ? step.execution_result.stdout : step.execution_result.stderr}
                          </Typography>
                        </Paper>
                      )}
                    </Box>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </Box>
        )}

        {/* Live Logs */}
        <Box
          sx={{
            height: 200,
            overflowY: 'auto',
            p: 2,
            backgroundColor: '#1e1e1e',
            borderTop: '1px solid #333'
          }}
        >
          <Typography variant="subtitle2" sx={{ color: '#888', mb: 1 }}>
            📋 Live Agent Logs:
          </Typography>
          {logs.map((log, index) => (
            <Fade in={true} timeout={500} key={`${log.timestamp}-${index}`}>
              <Box sx={{ mb: 0.5, display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: '#666',
                    minWidth: '60px',
                    fontFamily: 'monospace'
                  }}
                >
                  {log.timestamp}
                </Typography>
                <Typography
                  sx={{
                    color: getLevelColor(log.level),
                    fontFamily: 'monospace',
                    fontSize: '0.75rem'
                  }}
                >
                  {log.message}
                </Typography>
              </Box>
            </Fade>
          ))}
          <div ref={logsEndRef} />
        </Box>
      </Paper>
      
      {/* Code Explanation Dialog */}
      <Dialog
        open={codeExplanationOpen}
        onClose={() => setCodeExplanationOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Code Explanation: {selectedCode?.reasoning}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2 }}>
            This code step performs: <strong>{selectedCode?.reasoning.toLowerCase()}</strong>
          </Typography>
          
          <Typography variant="h6" sx={{ mb: 1 }}>
            What this code does:
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            • Uses standard Python libraries for data processing<br/>
            • Includes comprehensive error handling<br/>
            • Prints progress messages for monitoring<br/>
            • Follows best practices for data manipulation<br/>
            • Creates output files for the next processing step
          </Typography>
          
          <Typography variant="h6" sx={{ mb: 1 }}>
            Generated Code:
          </Typography>
          <Paper sx={{ p: 2, backgroundColor: '#f5f5f5', maxHeight: 300, overflowY: 'auto' }}>
            <Typography
              component="pre"
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                whiteSpace: 'pre-wrap',
                margin: 0,
              }}
            >
              {selectedCode?.code}
            </Typography>
          </Paper>
          
          {selectedCode?.execution_result && (
            <>
              <Typography variant="h6" sx={{ mb: 1, mt: 2 }}>
                Execution Result:
              </Typography>
              <Paper sx={{ 
                p: 2, 
                backgroundColor: selectedCode.execution_result.success ? '#e8f5e8' : '#fdeaea',
                maxHeight: 200, 
                overflowY: 'auto' 
              }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  Status: {selectedCode.execution_result.success ? '✅ Success' : '❌ Failed'}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  Execution Time: {selectedCode.execution_result.execution_time.toFixed(2)}s
                </Typography>
                <Typography
                  component="pre"
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    whiteSpace: 'pre-wrap',
                    margin: 0,
                  }}
                >
                  {selectedCode.execution_result.success ? 
                    selectedCode.execution_result.stdout : 
                    selectedCode.execution_result.stderr}
                </Typography>
              </Paper>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCodeExplanationOpen(false)} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Collapse>
  );
};

export default LLMAgentTerminal;
