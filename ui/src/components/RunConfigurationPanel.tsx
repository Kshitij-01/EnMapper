import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Stack,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Switch,
  FormControlLabel,
  Slider,
  Alert,
  Divider,
  Tooltip,
  IconButton,
  Grid,
} from '@mui/material';
import {
  Info as InfoIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  AccountBalance as BudgetIcon,
  Timeline as ModeIcon,
  AttachMoney as CostIcon,
} from '@mui/icons-material';

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

interface RunConfigurationPanelProps {
  config: RunConfiguration;
  onChange: (config: RunConfiguration) => void;
  onEstimateCost: () => void;
  estimatedCost: any;
}

const RunConfigurationPanel: React.FC<RunConfigurationPanelProps> = ({
  config,
  onChange,
  onEstimateCost,
  estimatedCost,
}) => {
  const handleConfigChange = (field: keyof RunConfiguration, value: any) => {
    onChange({ ...config, [field]: value });
  };

  const handleBudgetChange = (field: keyof RunConfiguration['budgetCaps'], value: number) => {
    onChange({
      ...config,
      budgetCaps: {
        ...config.budgetCaps,
        [field]: value,
      },
    });
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  };

  const modeDescriptions = {
    metadata_only: 'Analyze schema, column names, and data types without processing actual data content',
    data_mode: 'Full analysis including data patterns, samples, and content-based insights',
  };

  const laneDescriptions = {
    interactive: 'Fast processing with immediate feedback, suitable for exploratory analysis',
    flex: 'Balanced processing speed and resource usage for most production workloads',
    batch: 'Optimized for large datasets with longer processing times but lower costs',
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
          <ModeIcon sx={{ mr: 1 }} />
          Advanced Configuration
        </Typography>

        <Grid container spacing={3}>
          {/* Processing Mode */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                Processing Mode
                <Tooltip title="Choose how much of your data to analyze">
                  <IconButton size="small" sx={{ ml: 0.5 }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>
              
              <FormControl fullWidth size="small">
                <Select
                  value={config.mode}
                  onChange={(e) => handleConfigChange('mode', e.target.value)}
                >
                  <MenuItem value="metadata_only">
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Metadata Only
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Schema analysis only
                      </Typography>
                    </Box>
                  </MenuItem>
                  <MenuItem value="data_mode">
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Data Mode
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Full data analysis
                      </Typography>
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
              
              <Alert severity="info" sx={{ mt: 1 }}>
                <Typography variant="caption">
                  {modeDescriptions[config.mode]}
                </Typography>
              </Alert>
            </Box>
          </Grid>

          {/* Processing Lane */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                Processing Lane
                <Tooltip title="Choose processing speed vs cost trade-off">
                  <IconButton size="small" sx={{ ml: 0.5 }}>
                    <SpeedIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>
              
              <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
                {(['interactive', 'flex', 'batch'] as const).map((lane) => (
                  <Chip
                    key={lane}
                    label={lane.charAt(0).toUpperCase() + lane.slice(1)}
                    onClick={() => handleConfigChange('laneHint', lane)}
                    color={config.laneHint === lane ? 'primary' : 'default'}
                    variant={config.laneHint === lane ? 'filled' : 'outlined'}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Stack>
              
              <Alert severity="info">
                <Typography variant="caption">
                  {laneDescriptions[config.laneHint]}
                </Typography>
              </Alert>
            </Box>
          </Grid>

          {/* PII Masking */}
          <Grid size={{ xs: 12 }}>
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                Privacy & Security
                <SecurityIcon sx={{ ml: 0.5, fontSize: 16 }} />
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={config.piiMasking}
                    onChange={(e) => handleConfigChange('piiMasking', e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2" fontWeight={500}>
                      Enable PII Masking
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Automatically detect and mask personally identifiable information
                    </Typography>
                  </Box>
                }
              />
              
              <Alert 
                severity={config.piiMasking ? "success" : "warning"} 
                sx={{ mt: 1 }}
              >
                <Typography variant="caption">
                  {config.piiMasking 
                    ? "PII will be automatically detected and masked before any external processing"
                    : "Warning: PII masking is disabled. Ensure no sensitive data is processed."
                  }
                </Typography>
              </Alert>
            </Box>
          </Grid>

          {/* Budget Caps */}
          <Grid size={{ xs: 12 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
              Budget Limits
              <BudgetIcon sx={{ ml: 0.5, fontSize: 16 }} />
            </Typography>
            
            <Grid container spacing={2}>
              {/* Token Budget */}
              <Grid size={{ xs: 12, sm: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Token Limit: {config.budgetCaps.tokens.toLocaleString()}
                </Typography>
                <Slider
                  value={config.budgetCaps.tokens}
                  onChange={(_, value) => handleBudgetChange('tokens', value as number)}
                  min={1000}
                  max={1000000}
                  step={1000}
                  marks={[
                    { value: 1000, label: '1K' },
                    { value: 100000, label: '100K' },
                  ]}
                  valueLabelDisplay="auto"
                  scale={(x) => x}
                  valueLabelFormat={(value) => `${(value / 1000).toFixed(0)}K`}
                  sx={{
                    mb: 4,
                    '& .MuiSlider-markLabel': {
                      whiteSpace: 'nowrap',
                      fontSize: '0.75rem',
                    }
                  }}
                />
              </Grid>

              {/* USD Budget */}
              <Grid size={{ xs: 12, sm: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Cost Limit: ${config.budgetCaps.usd.toFixed(0)}
                </Typography>
                <Slider
                  value={config.budgetCaps.usd}
                  onChange={(_, value) => handleBudgetChange('usd', value as number)}
                  min={1}
                  max={1000}
                  step={1}
                  marks={[
                    { value: 1, label: '$1' },
                    { value: 1000, label: '$1K' },
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `$${value}`}
                  sx={{
                    mb: 4,
                    '& .MuiSlider-markLabel': {
                      whiteSpace: 'nowrap',
                      fontSize: '0.75rem',
                    }
                  }}
                />
              </Grid>

              {/* Time Budget */}
              <Grid size={{ xs: 12, sm: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Time Limit: {formatTime(config.budgetCaps.wallTimeS)}
                </Typography>
                <Slider
                  value={config.budgetCaps.wallTimeS}
                  onChange={(_, value) => handleBudgetChange('wallTimeS', value as number)}
                  min={300}
                  max={86400}
                  step={300}
                  marks={[
                    { value: 300, label: '5m' },
                    { value: 86400, label: '24h' },
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => formatTime(value)}
                  sx={{
                    mb: 4,
                    '& .MuiSlider-markLabel': {
                      whiteSpace: 'nowrap',
                      fontSize: '0.75rem',
                    }
                  }}
                />
              </Grid>
            </Grid>
          </Grid>

          {/* Cost Estimation */}
          {estimatedCost && (
            <Grid size={{ xs: 12 }}>
              <Divider sx={{ my: 2 }} />
              <Alert severity="info" icon={<CostIcon />}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Cost Estimation
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      Estimated Cost
                    </Typography>
                    <Typography variant="h6" color="primary">
                      ${Number(estimatedCost.estimated_cost_usd ?? estimatedCost.estimatedCostUsd ?? 0).toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      Tokens
                    </Typography>
                    <Typography variant="h6">
                      {Number(estimatedCost.estimated_tokens ?? estimatedCost.estimatedTokens ?? 0).toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      Processing Time
                    </Typography>
                    <Typography variant="h6">
                      ~{formatTime(Number(estimatedCost.processing_time_estimate_s ?? estimatedCost.processingTimeEstimateS ?? 0))}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence
                    </Typography>
                    <Typography variant="h6">
                      {estimatedCost.confidence ?? 'unknown'}
                    </Typography>
                  </Grid>
                </Grid>
              </Alert>
            </Grid>
          )}
        </Grid>

        <Divider sx={{ my: 3 }} />

        <Stack direction="row" spacing={2} justifyContent="flex-end" className="advanced-actions">
          <Button
            color="primary"
            variant="contained"
            disableElevation
            size="medium"
            disabled={false}
            onClick={onEstimateCost}
            startIcon={<CostIcon />}
          >
            Update Cost Estimate
          </Button>
          <Button
            variant="text"
            onClick={() => {
              // Reset to defaults
              onChange({
                mode: 'metadata_only',
                laneHint: 'interactive',
                piiMasking: true,
                budgetCaps: {
                  tokens: 100000,
                  usd: 10.0,
                  wallTimeS: 3600,
                },
              });
            }}
          >
            Reset to Defaults
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default RunConfigurationPanel;
