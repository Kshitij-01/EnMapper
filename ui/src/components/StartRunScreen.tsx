import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  Stack,
  Divider,
  IconButton,
  Tooltip,
  Alert,
  Collapse,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Storage as DatabaseIcon,
  Settings as SettingsIcon,
  PlayArrow as RunIcon,
  Info as InfoIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import FileUploadCard from './FileUploadCard';
import SQLConnectionCard from './SQLConnectionCard';
import RunConfigurationPanel from './RunConfigurationPanel';
import LLMAgentTerminal from './LLMAgentTerminal';
import DomainMappingProgress from './DomainMappingProgress';
import { EnMapperAPI, type DataSource } from '../services/api';
import { StartRunState } from '../App';

interface RunConfiguration {
  mode: 'metadata_only' | 'data_mode';
  laneHint: 'interactive' | 'flex' | 'batch';
  piiMasking: boolean;
  budgetCaps: {
    tokens: number;
    usd: number;
    wallTimeS: number;
  };
}

interface StartRunScreenProps {
  onNotification?: (notification: { type: 'success' | 'error' | 'warning' | 'info'; message: string; details?: string }) => void;
  onLoading?: (loading: boolean, operation?: string, progress?: number) => void;
  onError?: (error: Error | string, details?: string) => void;
  isLoading?: boolean;
  onDomainMappingComplete?: (results: any[]) => void;
  persistentState?: StartRunState;
  onStateChange?: (state: StartRunState) => void;
}

const StartRunScreen: React.FC<StartRunScreenProps> = ({ 
  onNotification, 
  onLoading, 
  onError, 
  isLoading,
  onDomainMappingComplete,
  persistentState,
  onStateChange
}) => {
  const [selectedSource, setSelectedSource] = useState<'file' | 'sql' | null>(null);
  const [runConfig, setRunConfig] = useState<RunConfiguration>({
    mode: 'metadata_only',
    laneHint: 'interactive',
    piiMasking: true,
    budgetCaps: {
      tokens: 100000,
      usd: 10.0,
      wallTimeS: 3600,
    },
  });
  const [estimatedCost, setEstimatedCost] = useState<any>(null);
  const [isCreatingRun, setIsCreatingRun] = useState(false);
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(true);
  const [domainMappingResults, setDomainMappingResults] = useState<any[]>([]);

  // Local state for immediate updates (syncs with persistent state)
  const [localCurrentRunId, setLocalCurrentRunId] = useState<string | null>(persistentState?.currentRunId || null);
  const [localShowLLMAgent, setLocalShowLLMAgent] = useState(persistentState?.showLLMAgent || false);
  const [localShowDomainMapping, setLocalShowDomainMapping] = useState(persistentState?.showDomainMapping || false);
  
  // Use local state for immediate responsiveness, persistent state for persistence
  const showLLMAgentTerminal = localShowLLMAgent;
  const showDomainMapping = localShowDomainMapping;
  const currentRunId = localCurrentRunId;
  const uploadedFile = persistentState?.uploadedFile || null;
  const isProcessing = persistentState?.isProcessing || false;

  // Sync local state with persistent state changes (but never set currentRunId to null once it's set)
  useEffect(() => {
    if (persistentState?.currentRunId && persistentState.currentRunId !== localCurrentRunId) {
      console.log('🔄 Syncing currentRunId from persistent state:', persistentState.currentRunId);
      setLocalCurrentRunId(persistentState.currentRunId);
    } else if (!persistentState?.currentRunId && localCurrentRunId) {
      console.log('⚠️ Persistent state trying to set currentRunId to null, but local has value:', localCurrentRunId, '- keeping local value');
      // Don't override with null if we have a local value
    }
    
    if (persistentState?.showLLMAgent !== localShowLLMAgent) {
      const nextShowLLM = !!persistentState?.showLLMAgent;
      console.log('🔄 Syncing showLLMAgent from persistent state:', nextShowLLM);
      setLocalShowLLMAgent(nextShowLLM);
    }
    if (persistentState?.showDomainMapping !== localShowDomainMapping) {
      const nextShowDomain = !!persistentState?.showDomainMapping;
      console.log('🔄 Syncing showDomainMapping from persistent state:', nextShowDomain);
      setLocalShowDomainMapping(nextShowDomain);
    }
  }, [persistentState, localCurrentRunId, localShowLLMAgent, localShowDomainMapping]);

  // Helper function to update persistent state
  const updatePersistentState = (updates: Partial<StartRunState>) => {
    if (onStateChange) {
      const currentState = persistentState || {
        uploadedFile: null,
        currentRunId: null,
        llmAgentLogs: [],
        domainMappingLogs: [],
        showLLMAgent: false,
        showDomainMapping: false,
        isProcessing: false
      };
      onStateChange({ ...currentState, ...updates });
    }
  };

  // Debug logging for LLM Agent Terminal visibility
  useEffect(() => {
    console.log('🔍 LLM Agent Terminal state changed:', { 
      showLLMAgentTerminal, 
      currentRunId, 
      shouldRender: showLLMAgentTerminal && currentRunId 
    });
  }, [showLLMAgentTerminal, currentRunId]);

  const setShowLLMAgentTerminal = (show: boolean) => {
    console.log('📺 setShowLLMAgentTerminal called with:', show);
    setLocalShowLLMAgent(show); // Immediate local update
    updatePersistentState({ showLLMAgent: show }); // Persistent update
  };
  
  const setShowDomainMapping = (show: boolean) => {
    console.log('🗺️ setShowDomainMapping called with:', show);
    setLocalShowDomainMapping(show); // Immediate local update
    updatePersistentState({ showDomainMapping: show }); // Persistent update
  };
  
  const setCurrentRunId = (runId: string | null) => {
    console.log('🆔 setCurrentRunId called with:', runId);
    setLocalCurrentRunId(runId); // Immediate local update
    updatePersistentState({ currentRunId: runId }); // Persistent update
  };
  
  const setUploadedFile = (file: any) => updatePersistentState({ uploadedFile: file });
  const setIsProcessing = (processing: boolean) => updatePersistentState({ isProcessing: processing });

  const handleSourceChange = (source: 'file' | 'sql') => {
    setSelectedSource(source);
  };

  const handleCreateRun = async () => {
    setIsCreatingRun(true);
    // TODO: Implement run creation logic
    console.log('Creating run with config:', runConfig);
    setIsCreatingRun(false);
  };

  const handleEstimateCost = async () => {
    try {
      onLoading?.(true, 'Estimating cost');

      // Build a minimal data source; fall back to local estimate if none selected
      let dataSource: DataSource | null = null;
      if (selectedSource === 'sql') {
        dataSource = { type: 'sql', location: 'connection', connection_params: {} } as DataSource;
      } else if (selectedSource === 'file') {
        dataSource = { type: 'file', location: 'uploaded' } as DataSource;
      }

      if (dataSource) {
        const response = await EnMapperAPI.estimateCost({
          data_source: dataSource,
          mode: runConfig.mode,
          sample_size: 1000,
        });
        setEstimatedCost(response);
        onNotification?.({ type: 'info', message: 'Cost estimate updated' });
      } else {
        // Local heuristic estimate when no source is selected
        const estimated_tokens = Math.min(runConfig.budgetCaps.tokens, 200000);
        const pricePerToken = 0.002 / 1000; // ~$2 / 1M tokens heuristic
        const estimated_cost_usd = Math.min(
          runConfig.budgetCaps.usd,
          Number((estimated_tokens * pricePerToken).toFixed(2))
        );
        const processing_time_estimate_s = Math.min(
          runConfig.budgetCaps.wallTimeS,
          Math.max(60, Math.round(estimated_tokens / 2500))
        );
        setEstimatedCost({
          estimated_tokens,
          estimated_cost_usd,
          processing_time_estimate_s,
          breakdown: {},
          confidence: 'low',
        } as any);
        onNotification?.({ type: 'warning', message: 'Local estimate shown (select a source for server estimate)' });
      }
    } catch (error: any) {
      onNotification?.({ type: 'error', message: 'Failed to estimate cost', details: error?.message });
    } finally {
      onLoading?.(false);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography
          variant="h1"
          sx={{
            color: 'white',
            mb: 2,
            fontWeight: 700,
          }}
        >
          EnMapper
        </Typography>
        <Typography
          variant="h5"
          sx={{
            color: '#000000',
            mb: 1,
            fontWeight: 400,
          }}
        >
          AI-Powered Data Mapping and Migration Platform
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: '#000000',
            maxWidth: 600,
            mx: 'auto',
          }}
        >
          Upload your data files or connect to SQL databases to automatically generate 
          schema mappings, detect patterns, and streamline your data migration process.
        </Typography>
      </Box>

      {/* Context Capsules */}
      <Box sx={{ mb: 4 }}>
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
              flexWrap="wrap"
            >
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="h6" color="text.secondary">
                  Current Configuration:
                </Typography>
                <Chip
                  icon={<TimelineIcon />}
                  label={runConfig.mode === 'metadata_only' ? 'Metadata Only' : 'Data Mode'}
                  color="primary"
                  variant="outlined"
                />
                <Chip
                  icon={<SpeedIcon />}
                  label={runConfig.laneHint.charAt(0).toUpperCase() + runConfig.laneHint.slice(1)}
                  color="secondary"
                  variant="outlined"
                />
                <Chip
                  icon={<SecurityIcon />}
                  label={runConfig.piiMasking ? 'PII Masking ON' : 'PII Masking OFF'}
                  color={runConfig.piiMasking ? 'success' : 'warning'}
                  variant="outlined"
                />
                <Chip
                  label={`Budget: $${runConfig.budgetCaps.usd}`}
                  color="info"
                  variant="outlined"
                />
              </Stack>
              <Stack direction="row" spacing={1}>
                <Tooltip title="Advanced Configuration">
                  <IconButton
                    onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                    color={showAdvancedConfig ? 'primary' : 'default'}
                  >
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Help & Documentation">
                  <IconButton color="default">
                    <InfoIcon />
                  </IconButton>
                </Tooltip>
              </Stack>
            </Stack>
          </CardContent>
        </Card>

        {/* Advanced Configuration Panel */}
        <Collapse in={showAdvancedConfig}>
          <RunConfigurationPanel
            config={runConfig}
            onChange={setRunConfig}
            onEstimateCost={handleEstimateCost}
            estimatedCost={estimatedCost}
          />
        </Collapse>
      </Box>

      {/* Main Content */}
      <Grid container spacing={4}>
        {/* Data Source Selection */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Typography variant="h4" sx={{ mb: 3, color: 'white', fontWeight: 600 }}>
            Choose Your Data Source
          </Typography>
          
          <Grid container spacing={3}>
            {/* File Upload Card */}
            <Grid size={{ xs: 12, md: 6 }}>
              <FileUploadCard
                selected={selectedSource === 'file'}
                onSelect={() => handleSourceChange('file')}
                onNotification={onNotification}
                onLoading={onLoading}
                onError={onError}
                agentMode={true}
                onFileUploaded={(result) => {
                  console.log('🎉 onFileUploaded called with result:', result);
                  onNotification?.({
                    type: 'success',
                    message: 'File uploaded successfully!',
                    details: `Run created: ${result.run_id}`
                  });
                  
                  // Always set the current run ID
                  console.log('📝 Setting current run ID:', result.run_id);
                  setCurrentRunId(result.run_id);
                  setIsProcessing(true);
                  
                  // Always open the LLM Agent terminal after upload
                  console.log('🖥️ Setting LLM Agent Terminal to visible');
                  setShowLLMAgentTerminal(true);
                  console.log('🎬 LLM Agent Terminal visibility state updated');
                  onNotification?.({
                    type: 'info',
                    message: `LLM Agent processing started for run: ${result.run_id}`,
                    details: 'Check the LLM Agent Terminal below for live code generation and execution'
                  });
                }}
              />
            </Grid>

            {/* SQL Connection Card */}
            <Grid size={{ xs: 12, md: 6 }}>
              <SQLConnectionCard
                selected={selectedSource === 'sql'}
                onSelect={() => handleSourceChange('sql')}
              />
            </Grid>
          </Grid>

          {/* File Format Support Info */}
          <Alert
            severity="info"
            sx={{ mt: 3, backgroundColor: 'rgba(255, 255, 255, 0.95)' }}
          >
            <Typography variant="body2">
              <strong>Supported formats:</strong> CSV, TSV, Parquet, JSON, JSONL files up to 500MB. 
              SQL databases: PostgreSQL, MySQL, SQLite with read-only access recommended.
            </Typography>
          </Alert>
        </Grid>

        {/* Action Panel */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Card sx={{ position: 'sticky', top: 20 }}>
            <CardContent>
              <Typography variant="h5" sx={{ mb: 2, textAlign: 'center' }}>
                Start Processing
              </Typography>
              
              <Divider sx={{ mb: 2 }} />
              
              <Stack spacing={2}>
                {/* Cost Estimation */}
                {estimatedCost && (
                  <Alert severity="info">
                    <Typography variant="body2">
                      <strong>Estimated Cost:</strong> ${estimatedCost.estimated_cost_usd}<br />
                      <strong>Processing Time:</strong> ~{estimatedCost.processing_time_estimate_s}s<br />
                      <strong>Tokens:</strong> {estimatedCost.estimated_tokens}
                    </Typography>
                  </Alert>
                )}

                {/* Instructions */}
                <Alert severity="info" sx={{ width: '100%' }}>
                  <Typography variant="body2">
                    <strong>Step 1:</strong> Upload a file or connect to database<br/>
                    <strong>Step 2:</strong> Use LLM Agent for data processing (optional)<br/>
                    <strong>Step 3:</strong> Run Domain Mapping to assign domains
                  </Typography>
                </Alert>

                {/* Cost Estimation Button */}
                <Button
                  variant="outlined"
                  fullWidth
                  disabled={!selectedSource}
                  onClick={handleEstimateCost}
                  sx={{ py: 1 }}
                >
                  Get Cost Estimate
                </Button>

                {/* Test Agent Button */}
                <Button
                  variant="contained"
                  fullWidth
                  color="secondary"
                  disabled={!currentRunId}
                  onClick={async () => {
                    if (!currentRunId) return;
                    
                    setShowLLMAgentTerminal(true);
                    
                    onNotification?.({
                      type: 'info',
                      message: `Testing LLM agent with run: ${currentRunId}`,
                      details: 'LLM Agent Terminal will show live code generation and execution'
                    });
                  }}
                  sx={{ py: 1 }}
                >
                  🧠 Test LLM Agent {currentRunId ? `(${currentRunId.slice(0, 8)})` : '(Upload file first)'}
                </Button>
              </Stack>

              <Divider sx={{ my: 2 }} />

              <Typography variant="body2" color="text.secondary" textAlign="center">
                Processing will begin once you start the run. You'll be able to monitor 
                progress and view results in real-time.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* LLM Agent Terminal */}
      {showLLMAgentTerminal && currentRunId && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, color: 'primary.main' }}>
            🤖 LLM Agent Terminal - Processing Run: {currentRunId.slice(0, 8)}
          </Typography>
          <LLMAgentTerminal
            runId={currentRunId}
            isVisible={showLLMAgentTerminal}
            onClose={() => setShowLLMAgentTerminal(false)}
            onComplete={() => {
              // Auto-trigger domain mapping after LLM completes
              setIsProcessing(false);
              onNotification?.({
                type: 'success',
                message: 'LLM Agent completed! Starting Domain Mapping...',
                details: 'The system will now automatically map your data to domain categories'
              });
              // Keep LLM agent visible and add domain mapping below
              setShowDomainMapping(true);
            }}
          />
        </Box>
      )}
      
      {/* Domain Mapping Progress - shows after LLM Agent */}
      {showDomainMapping && currentRunId && (
        <DomainMappingProgress
          runId={currentRunId}
          isVisible={showDomainMapping}
          onClose={() => setShowDomainMapping(false)}
          onComplete={(mappings) => {
            // Store results and show completion notification
            setDomainMappingResults(mappings);
            setIsProcessing(false);
            onNotification?.({
              type: 'success',
              message: `Domain mapping completed with ${mappings.length} columns mapped!`,
              details: 'Results have been transferred to Domain Studio tab for review and editing'
            });
            // Pass results to parent (App.tsx)
            if (onDomainMappingComplete) {
              onDomainMappingComplete(mappings);
            }
          }}
        />
      )}

      {/* Show status after file upload */}
      {currentRunId && !showLLMAgentTerminal && !showDomainMapping && (
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Alert severity="info" sx={{ mb: 2, maxWidth: 600, mx: 'auto' }}>
            <Typography variant="body2">
              File uploaded successfully! Run ID: <strong>{currentRunId.slice(0, 8)}</strong>
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
              You can now use the "Test LLM Agent" button above or proceed directly to Domain Mapping.
            </Typography>
          </Alert>
          
          <Button
            variant="contained"
            size="large"
            onClick={() => setShowDomainMapping(true)}
            startIcon={<TimelineIcon />}
            sx={{
              background: 'linear-gradient(45deg, #9c27b0 30%, #673ab7 90%)',
              color: 'white',
              py: 1.5,
              px: 4,
            }}
          >
            Start Domain Mapping Process
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default StartRunScreen;
export { StartRunScreen };
