import React, { useState } from 'react';
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
import { EnMapperAPI, type DataSource } from '../services/api';

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
}

const StartRunScreen: React.FC<StartRunScreenProps> = ({ 
  onNotification, 
  onLoading, 
  onError, 
  isLoading 
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
            color: 'rgba(255, 255, 255, 0.9)',
            mb: 1,
            fontWeight: 400,
          }}
        >
          AI-Powered Data Mapping and Migration Platform
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: 'rgba(255, 255, 255, 0.8)',
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
                onFileUploaded={(result) => {
                  onNotification?.({
                    type: 'success',
                    message: 'File uploaded successfully!',
                    details: `Run created: ${result.run_id}`
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

                {/* Run Button */}
                <Button
                  variant="contained"
                  size="large"
                  fullWidth
                  disabled={!selectedSource || isCreatingRun}
                  onClick={handleCreateRun}
                  startIcon={<RunIcon />}
                  sx={{
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                  }}
                >
                  {isCreatingRun ? 'Creating Run...' : 'Start Data Processing'}
                </Button>

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
    </Box>
  );
};

export default StartRunScreen;
export { StartRunScreen };
