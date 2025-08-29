import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Chip,
  Stack,
  Collapse,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Fade,
  CircularProgress,
  Button,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Schema as SchemaIcon,
  Assignment as AssignmentIcon,
  Analytics as AnalyticsIcon,
  PlayArrow as StartIcon,
} from '@mui/icons-material';

interface ColumnMapping {
  columnName: string;
  sourceType: string;
  detectedDomain: string;
  confidence: number;
  confidenceBand: 'HIGH' | 'MEDIUM' | 'LOW';
  evidence?: {
    nameScore: number;
    patternScore: number;
    valueScore: number;
    unitScore: number;
  };
  status: 'pending' | 'processing' | 'completed' | 'failed';
  alternativeDomains?: Array<{
    domain: string;
    confidence: number;
  }>;
}

interface MappingStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
  progress?: number;
  details?: string;
  startTime?: Date;
  endTime?: Date;
  metrics?: {
    totalColumns?: number;
    processedColumns?: number;
    highConfidence?: number;
    mediumConfidence?: number;
    lowConfidence?: number;
  };
}

interface DomainMappingProgressProps {
  runId: string;
  isVisible: boolean;
  onClose?: () => void;
  onComplete?: (mappings: ColumnMapping[]) => void;
}

const DomainMappingProgress: React.FC<DomainMappingProgressProps> = ({
  runId,
  isVisible,
  onClose,
  onComplete,
}) => {
  const [steps, setSteps] = useState<MappingStep[]>([
    {
      id: 'analyze',
      name: 'Analyze Schema',
      description: 'Analyzing column names, types, and sample values',
      status: 'pending',
    },
    {
      id: 'generate',
      name: 'Generate Candidates',
      description: 'Finding potential domain matches using AI',
      status: 'pending',
    },
    {
      id: 'score',
      name: 'Score & Rank',
      description: 'Calculating confidence scores for each mapping',
      status: 'pending',
    },
    {
      id: 'validate',
      name: 'Validate Results',
      description: 'Checking mappings against business rules',
      status: 'pending',
    },
  ]);

  const [mappings, setMappings] = useState<ColumnMapping[]>([]);
  const [activeStep, setActiveStep] = useState(0);
  const [overallProgress, setOverallProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
  const eventSourceRef = useRef<EventSource | null>(null);

  const toggleColumnExpand = (columnName: string) => {
    setExpandedColumns(prev => {
      const newSet = new Set(prev);
      if (newSet.has(columnName)) {
        newSet.delete(columnName);
      } else {
        newSet.add(columnName);
      }
      return newSet;
    });
  };

  const startDomainMapping = async () => {
    if (!runId) return;
    
    setIsProcessing(true);
    setActiveStep(0);
    setOverallProgress(0);
    
    // Reset steps
    setSteps(prev => prev.map(step => ({ ...step, status: 'pending', progress: 0 })));
    
    try {
      // Connect to domain mapping SSE endpoint
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      
      const eventSource = new EventSource(`http://localhost:8000/api/v1/domains/mapping-stream/${runId}`);
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMappingEvent(data);
        } catch (error) {
          console.error('Failed to parse mapping event:', error);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('Domain mapping stream error:', error);
        setIsProcessing(false);
        eventSource.close();
        eventSourceRef.current = null;
      };
      
    } catch (error) {
      console.error('Failed to start domain mapping:', error);
      setIsProcessing(false);
    }
  };

  // Auto-start when component becomes visible
  useEffect(() => {
    if (isVisible && runId && !isProcessing) {
      startDomainMapping();
    }
    
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [isVisible, runId]);

  const handleMappingEvent = (event: any) => {
    const { type, data } = event;
    
    switch (type) {
      case 'step_start':
        updateStep(data.stepId, { status: 'active', startTime: new Date() });
        break;
        
      case 'step_progress':
        updateStep(data.stepId, { progress: data.progress, details: data.details });
        break;
        
      case 'step_complete':
        updateStep(data.stepId, { 
          status: 'completed', 
          endTime: new Date(), 
          metrics: data.metrics 
        });
        setActiveStep(prev => prev + 1);
        break;
        
      case 'column_mapped':
        addOrUpdateMapping(data.mapping);
        break;
        
      case 'overall_progress':
        setOverallProgress(data.progress);
        break;
        
      case 'mapping_complete':
        setIsProcessing(false);
        if (onComplete) {
          onComplete(mappings);
        }
        break;
        
      case 'error':
        console.error('Domain mapping error:', data.error);
        updateStep(data.stepId || 'analyze', { status: 'failed', details: data.error });
        setIsProcessing(false);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        break;
    }
  };

  const updateStep = (stepId: string, updates: Partial<MappingStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, ...updates } : step
    ));
  };

  const addOrUpdateMapping = (mapping: ColumnMapping) => {
    setMappings(prev => {
      const index = prev.findIndex(m => m.columnName === mapping.columnName);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = mapping;
        return updated;
      }
      return [...prev, mapping];
    });
  };

  const getConfidenceColor = (band: string) => {
    switch (band) {
      case 'HIGH': return '#4caf50';
      case 'MEDIUM': return '#ff9800';
      case 'LOW': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const getStepIcon = (step: MappingStep) => {
    if (step.status === 'completed') {
      return <CheckIcon sx={{ color: '#4caf50' }} />;
    } else if (step.status === 'failed') {
      return <ErrorIcon sx={{ color: '#f44336' }} />;
    } else if (step.status === 'active') {
      return <CircularProgress size={20} />;
    }
    return null;
  };

  return (
    <Collapse in={isVisible}>
      <Paper
        elevation={4}
        sx={{
          mt: 2,
          backgroundColor: '#fafafa',
          border: '1px solid #e0e0e0',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 2,
            backgroundColor: 'primary.main',
            color: 'white',
          }}
        >
          <Stack direction="row" spacing={2} alignItems="center">
            <SchemaIcon />
            <Typography variant="h6" fontWeight={600}>
              Domain Mapping Engine
            </Typography>
            {isProcessing && (
              <Chip
                label="Processing"
                size="small"
                sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                icon={<CircularProgress size={16} sx={{ color: 'white' }} />}
              />
            )}
          </Stack>
          
          <Stack direction="row" spacing={1}>
            {!isProcessing && mappings.length === 0 && (
              <Button
                variant="contained"
                size="small"
                startIcon={<StartIcon />}
                onClick={startDomainMapping}
                sx={{
                  backgroundColor: 'white',
                  color: 'primary.main',
                  '&:hover': { backgroundColor: '#f5f5f5' },
                }}
              >
                Start Mapping
              </Button>
            )}
            {onClose && (
              <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}>
                <CollapseIcon />
              </IconButton>
            )}
          </Stack>
        </Box>

        {/* Overall Progress */}
        <Box sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
          <Stack spacing={1}>
            <Typography variant="body2" color="text.secondary">
              Overall Progress: {overallProgress}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={overallProgress}
              sx={{
                height: 8,
                borderRadius: 4,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 4,
                  background: 'linear-gradient(45deg, #2196f3 30%, #21cbf3 90%)',
                },
              }}
            />
          </Stack>
        </Box>

        {/* Process Steps */}
        <Box sx={{ p: 2 }}>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.id} completed={step.status === 'completed'}>
                <StepLabel
                  StepIconComponent={() => getStepIcon(step)}
                  sx={{
                    '& .MuiStepLabel-label': {
                      fontWeight: step.status === 'active' ? 600 : 400,
                    },
                  }}
                >
                  {step.name}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" color="text.secondary">
                    {step.description}
                  </Typography>
                  {step.details && (
                    <Alert severity="info" sx={{ mt: 1 }}>
                      {step.details}
                    </Alert>
                  )}
                  {step.progress !== undefined && step.status === 'active' && (
                    <LinearProgress
                      variant="determinate"
                      value={step.progress}
                      sx={{ mt: 1, mb: 1 }}
                    />
                  )}
                  {step.metrics && (
                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      <Chip
                        label={`Total: ${step.metrics.totalColumns}`}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={`High: ${step.metrics.highConfidence}`}
                        size="small"
                        sx={{ borderColor: '#4caf50', color: '#4caf50' }}
                        variant="outlined"
                      />
                      <Chip
                        label={`Medium: ${step.metrics.mediumConfidence}`}
                        size="small"
                        sx={{ borderColor: '#ff9800', color: '#ff9800' }}
                        variant="outlined"
                      />
                      <Chip
                        label={`Low: ${step.metrics.lowConfidence}`}
                        size="small"
                        sx={{ borderColor: '#f44336', color: '#f44336' }}
                        variant="outlined"
                      />
                    </Stack>
                  )}
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </Box>

        {/* Mapping Results Table */}
        {mappings.length > 0 && (
          <Box sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <AssignmentIcon />
              Column Mappings
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column Name</TableCell>
                    <TableCell>Source Type</TableCell>
                    <TableCell>Detected Domain</TableCell>
                    <TableCell align="center">Confidence</TableCell>
                    <TableCell align="center">Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {mappings.map((mapping) => (
                    <React.Fragment key={mapping.columnName}>
                      <TableRow
                        sx={{
                          '&:hover': { backgroundColor: '#f5f5f5' },
                          cursor: mapping.evidence ? 'pointer' : 'default',
                        }}
                      >
                        <TableCell sx={{ fontWeight: 500 }}>
                          {mapping.columnName}
                        </TableCell>
                        <TableCell>
                          <Chip label={mapping.sourceType} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {mapping.detectedDomain}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={`${(mapping.confidence * 100).toFixed(0)}%`}
                            size="small"
                            sx={{
                              backgroundColor: getConfidenceColor(mapping.confidenceBand) + '20',
                              color: getConfidenceColor(mapping.confidenceBand),
                              fontWeight: 600,
                            }}
                          />
                        </TableCell>
                        <TableCell align="center">
                          {mapping.evidence && (
                            <IconButton
                              size="small"
                              onClick={() => toggleColumnExpand(mapping.columnName)}
                            >
                              {expandedColumns.has(mapping.columnName) ? <CollapseIcon /> : <ExpandIcon />}
                            </IconButton>
                          )}
                        </TableCell>
                      </TableRow>
                      
                      {/* Expanded Details Row */}
                      <TableRow>
                        <TableCell colSpan={5} sx={{ py: 0 }}>
                          <Collapse in={expandedColumns.has(mapping.columnName)}>
                            <Box sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
                              <Stack spacing={2}>
                                {/* Evidence Scores */}
                                {mapping.evidence && (
                                  <Box>
                                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                      Evidence Scores:
                                    </Typography>
                                    <Stack direction="row" spacing={2}>
                                      <Chip
                                        label={`Name: ${(mapping.evidence.nameScore * 100).toFixed(0)}%`}
                                        size="small"
                                        variant="outlined"
                                      />
                                      <Chip
                                        label={`Pattern: ${(mapping.evidence.patternScore * 100).toFixed(0)}%`}
                                        size="small"
                                        variant="outlined"
                                      />
                                      <Chip
                                        label={`Values: ${(mapping.evidence.valueScore * 100).toFixed(0)}%`}
                                        size="small"
                                        variant="outlined"
                                      />
                                      <Chip
                                        label={`Units: ${(mapping.evidence.unitScore * 100).toFixed(0)}%`}
                                        size="small"
                                        variant="outlined"
                                      />
                                    </Stack>
                                  </Box>
                                )}
                                
                                {/* Alternative Domains */}
                                {mapping.alternativeDomains && mapping.alternativeDomains.length > 0 && (
                                  <Box>
                                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                      Alternative Domains:
                                    </Typography>
                                    <Stack direction="row" spacing={1}>
                                      {mapping.alternativeDomains.map((alt, idx) => (
                                        <Chip
                                          key={idx}
                                          label={`${alt.domain} (${(alt.confidence * 100).toFixed(0)}%)`}
                                          size="small"
                                          variant="outlined"
                                          sx={{ color: 'text.secondary' }}
                                        />
                                      ))}
                                    </Stack>
                                  </Box>
                                )}
                              </Stack>
                            </Box>
                          </Collapse>
                        </TableCell>
                      </TableRow>
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Paper>
    </Collapse>
  );
};

export default DomainMappingProgress;
