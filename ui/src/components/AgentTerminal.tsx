import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Chip,
  Stack,
  LinearProgress,
  Alert,
  Fade,
  Collapse,
} from '@mui/material';
import {
  Terminal as TerminalIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  PlayArrow as PlayIcon,
  SmartToy as AgentIcon,
} from '@mui/icons-material';

interface AgentLog {
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  step?: string;
}

interface AgentTerminalProps {
  runId: string;
  isVisible: boolean;
  onClose: () => void;
}

const AgentTerminal: React.FC<AgentTerminalProps> = ({ runId, isVisible, onClose }) => {
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [agentStatus, setAgentStatus] = useState<any>(null);
  const [isPolling, setIsPolling] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  const fetchAgentStatus = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/agent/status/${runId}`);
      if (response.ok) {
        const status = await response.json();
        setAgentStatus(status);
        
        // If agent is active, start polling for activity
        if (status.agent_active && !isPolling) {
          startPolling();
        } else if (!status.agent_active && isPolling) {
          stopPolling();
        }
      }
    } catch (error) {
      console.error('Failed to fetch agent status:', error);
    }
  };

  const fetchAgentLogs = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/agent/logs/${runId}`);
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs.map((log: any) => ({
          timestamp: new Date(log.timestamp).toLocaleTimeString(),
          level: log.level,
          message: log.message,
          step: log.step
        })));
      }
    } catch (error) {
      console.error('Failed to fetch agent logs:', error);
    }
  };

  const startPolling = () => {
    if (pollIntervalRef.current) return;
    
    setIsPolling(true);
    pollIntervalRef.current = setInterval(() => {
      fetchAgentStatus();
      fetchAgentLogs();
    }, 2000);
  };

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    setIsPolling(false);
  };

  const addSimulatedLogs = () => {
    // In a real implementation, these would come from the backend
    const simulatedSteps = [
      { step: 'extract', message: '📦 Extracting ZIP file...', level: 'info' as const },
      { step: 'list_files', message: '📋 Listing extracted files...', level: 'info' as const },
      { step: 'read_sample', message: '🔍 Reading data samples...', level: 'info' as const },
      { step: 'domain_assign', message: '🎯 Assigning semantic domains...', level: 'info' as const },
      { step: 'pii_scan', message: '🔒 Scanning for PII...', level: 'warning' as const },
      { step: 'python_exec', message: '🐍 Executing standardization code...', level: 'info' as const },
      { step: 'complete', message: '✅ Agent processing completed!', level: 'success' as const },
    ];

    // Add logs progressively
    if (logs.length < simulatedSteps.length) {
      const nextLog = simulatedSteps[logs.length];
      setLogs(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        ...nextLog
      }]);
    } else if (logs.length === simulatedSteps.length) {
      stopPolling();
    }
  };

  useEffect(() => {
    if (isVisible && runId) {
      fetchAgentStatus();
      fetchAgentLogs();
    }
    
    return () => {
      stopPolling();
    };
  }, [isVisible, runId]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'error': return '#f44336';
      default: return '#2196f3';
    }
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'error': return '❌';
      default: return '🔵';
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
          fontFamily: 'monospace',
          border: '1px solid #333',
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        {/* Terminal Header */}
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
              Agent Terminal - Run {runId.slice(0, 8)}
            </Typography>
            {isPolling && (
              <Chip
                icon={<PlayIcon />}
                label="Active"
                size="small"
                color="success"
                variant="outlined"
              />
            )}
          </Stack>
          
          <Stack direction="row" spacing={1}>
            <IconButton onClick={fetchAgentStatus} size="small" sx={{ color: '#fff' }}>
              <RefreshIcon />
            </IconButton>
            <IconButton onClick={onClose} size="small" sx={{ color: '#fff' }}>
              <CloseIcon />
            </IconButton>
          </Stack>
        </Box>

        {/* Agent Status */}
        {agentStatus && (
          <Box sx={{ p: 2, backgroundColor: '#252525' }}>
            <Stack direction="row" spacing={2} alignItems="center">
              <Typography variant="body2" sx={{ color: '#888' }}>
                Status:
              </Typography>
              <Chip
                label={agentStatus.agent_active ? 'Active' : 'Inactive'}
                size="small"
                color={agentStatus.agent_active ? 'success' : 'default'}
              />
              <Typography variant="body2" sx={{ color: '#888' }}>
                Workspace: {agentStatus.workspace_files?.length || 0} files
              </Typography>
            </Stack>
          </Box>
        )}

        {/* Loading Progress */}
        {isPolling && (
          <LinearProgress
            sx={{
              '& .MuiLinearProgress-bar': {
                backgroundColor: '#00ff00'
              }
            }}
          />
        )}

        {/* Terminal Output */}
        <Box
          sx={{
            height: 300,
            overflowY: 'auto',
            p: 2,
            backgroundColor: '#1e1e1e',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: '#2d2d2d',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#555',
              borderRadius: '4px',
            },
          }}
        >
          {logs.length === 0 ? (
            <Typography sx={{ color: '#888', fontStyle: 'italic' }}>
              Waiting for agent activity...
            </Typography>
          ) : (
            logs.map((log, index) => (
              <Fade in={true} timeout={500} key={index}>
                <Box sx={{ mb: 1, display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: '#666',
                      minWidth: '80px',
                      fontFamily: 'monospace'
                    }}
                  >
                    {log.timestamp}
                  </Typography>
                  <Typography
                    sx={{
                      color: getLevelColor(log.level),
                      fontFamily: 'monospace',
                      fontSize: '0.875rem'
                    }}
                  >
                    {getLevelIcon(log.level)} {log.message}
                  </Typography>
                </Box>
              </Fade>
            ))
          )}
          <div ref={logsEndRef} />
        </Box>

        {/* No Agent Alert */}
        {agentStatus && !agentStatus.agent_active && (
          <Alert severity="info" sx={{ m: 2 }}>
            No agent activity detected. Upload a file with Agent Mode enabled to see live processing.
          </Alert>
        )}
      </Paper>
    </Collapse>
  );
};

export default AgentTerminal;
