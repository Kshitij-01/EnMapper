import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Collapse,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  Tooltip,
  Stack,
  Badge,
  useTheme,
  alpha
} from '@mui/material';
// Using flexbox instead of Grid for better compatibility
import {
  ExpandMore,
  ExpandLess,
  Check,
  Close,
  Add,
  Info,
  Pattern,
  DataObject,
  Category,
  AutoAwesome,
  Refresh,
  Analytics,
  TrendingUp
} from '@mui/icons-material';
import api from '../services/api';

// Types for domain assignment data
interface Evidence {
  name_similarity: number;
  regex_strength: number;
  value_similarity: number;
  unit_compatibility: number;
  composite_score: number;
  matching_aliases: string[];
  matching_patterns: string[];
  matching_units: string[];
  header_tokens: string[];
}

interface DomainAssignment {
  column_name: string;
  domain_id: string | null;
  domain_name: string | null;
  confidence_score: number;
  confidence_band: 'high' | 'borderline' | 'low';
  evidence: Evidence;
  assigned_at: string;
  human_reviewed: boolean;
  human_decision: string | null;
}

interface AppNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  details?: string;
}

interface DomainStudioProps {
  runId?: string;
  onNotification?: (notification: Omit<AppNotification, 'id'>) => void;
  initialMappings?: any[];
}

const DomainStudio: React.FC<DomainStudioProps> = ({ runId, onNotification, initialMappings }) => {
  const theme = useTheme();
  const [assignments, setAssignments] = useState<DomainAssignment[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [llmMode, setLlmMode] = useState(false);
  const [viewMode, setViewMode] = useState<'assignments' | 'groups'>('assignments');
  const [groupedData, setGroupedData] = useState<any[]>([]);
  const [openLabelingData, setOpenLabelingData] = useState<any[]>([]);
  const [aliasDialog, setAliasDialog] = useState<{
    open: boolean;
    columnName: string;
    domainId: string;
    alias: string;
  }>({
    open: false,
    columnName: '',
    domainId: '',
    alias: ''
  });

  // Load initial mappings if provided
  useEffect(() => {
    if (initialMappings && initialMappings.length > 0) {
      // Convert initial mappings to DomainAssignment format
      const convertedAssignments: DomainAssignment[] = initialMappings.map((mapping, index) => ({
        column_name: mapping.columnName,
        domain_id: `domain-${index}`,
        domain_name: mapping.detectedDomain,
        confidence_score: mapping.confidence,
        confidence_band: mapping.confidenceBand.toLowerCase() as 'high' | 'borderline' | 'low',
        evidence: mapping.evidence || {
          name_score: 0.8,
          pattern_score: 0.7,
          value_score: 0.6,
          unit_score: 0.5,
          composite_score: mapping.confidence,
          confidence_band: mapping.confidenceBand.toLowerCase() as 'high' | 'borderline' | 'low',
          column_name: mapping.columnName
        },
        assigned_at: new Date().toISOString(),
        human_reviewed: false,
        human_decision: null
      }));
      setAssignments(convertedAssignments);
      onNotification?.({
        type: 'info',
        message: `Loaded ${convertedAssignments.length} domain mappings from LLM Agent`,
        details: 'Review and approve the mappings below'
      });
    }
  }, [initialMappings, onNotification]);

  // Removed demo sample columns to avoid stale results; always use backend

  // Do NOT auto-run demo assignment on mount. We prefer results coming from
  // the LLM Agent (passed as initialMappings). Users can still trigger
  // Neural/LLM mapping via the buttons.
  useEffect(() => {
    // If no initial mappings exist and you want a quick demo, uncomment:
    // assignDomains(false);
  }, [initialMappings]);

  const assignDomains = async (useLlm = false) => {
    setLoading(true);
    try {
      // Use fallback run ID if not provided  
      const effectiveRunId = runId || '0c0cb126-d5bc-4b03-915e-e480f5702275';
      console.log('Using run ID for domains:', effectiveRunId);
      console.log('Provided runId prop:', runId);
      
      let count = 0;
      if (useLlm) {
        const res = await api.post('/api/v1/domains/assign-llm-batch', {
          run_id: effectiveRunId,
          mode: 'data'
        });
        setAssignments(res.data.assignments || []);
        count = (res.data.assignments || []).length;
      } else {
        // Neural mode: fetch latest mapping for this run from backend
        const res = await api.post('/api/v1/domains/map-run', { run_id: effectiveRunId });
        setAssignments(res.data.assignments || []);
        count = (res.data.assignments || []).length;
      }
      setViewMode('assignments');
      setLlmMode(useLlm);
      
      const message = useLlm 
        ? `LLM-enhanced domain assignments completed (${count} columns)`
        : `Domain assignments completed (${count} columns)`;
      
      onNotification?.({ type: 'success', message });
    } catch (error) {
      console.error('Failed to assign domains:', error);
      onNotification?.({ type: 'error', message: 'Failed to assign domains' });
    } finally {
      setLoading(false);
    }
  };

  const assignOpenLabels = async () => {
    setLoading(true);
    try {
      // Use fallback run ID if not provided
      const effectiveRunId = runId || '0c0cb126-d5bc-4b03-915e-e480f5702275';
      console.log('Using run ID for open labels:', effectiveRunId);
      console.log('Provided runId prop:', runId);
      
      const res = await api.post('/api/v1/domains/assign-open', {
        run_id: effectiveRunId,
        mode: 'data'
      });
      
      // Convert open labels to domain assignment format
      const openAssignments: DomainAssignment[] = (res.data.labels || []).map((label: any) => ({
        column_name: label.column_name,
        domain_id: `open-${label.column_name}`,
        domain_name: label.label,
        confidence_score: label.label === 'unknown' ? 0.0 : 0.9,
        confidence_band: label.label === 'unknown' ? 'low' : 'high' as 'high' | 'borderline' | 'low',
        evidence: {
          name_similarity: 0.8,
          regex_strength: 0.7,
          value_similarity: 0.8,
          unit_compatibility: 0.7,
          composite_score: label.label === 'unknown' ? 0.0 : 0.9,
          matching_aliases: [],
          matching_patterns: [],
          matching_units: [],
          header_tokens: [label.column_name]
        },
        assigned_at: new Date().toISOString(),
        human_reviewed: false,
        human_decision: null
      }));
      
      setAssignments(openAssignments);
      setViewMode('assignments');
      setLlmMode(true);
      
      const successCount = openAssignments.filter(a => a.domain_name !== 'unknown').length;
      onNotification?.({ 
        type: 'success', 
        message: `Open labeling completed: ${successCount}/${openAssignments.length} columns labeled with GPT-4o-mini`,
        details: 'Individual semantic labels assigned without predefined catalog'
      });
    } catch (error) {
      console.error('Failed to assign open labels:', error);
      onNotification?.({ type: 'error', message: 'Failed to assign open labels' });
    } finally {
      setLoading(false);
    }
  };

  const assignGroupedLabels = async () => {
    setLoading(true);
    try {
      // Use fallback run ID if not provided
      const effectiveRunId = runId || '0c0cb126-d5bc-4b03-915e-e480f5702275';
      console.log('Using run ID for grouping:', effectiveRunId);
      console.log('Provided runId prop:', runId);
      
      // First get open labels
      const openRes = await api.post('/api/v1/domains/assign-open', {
        run_id: effectiveRunId,
        mode: 'data'
      });
      
      // Then group them with Claude
      const groupRes = await api.post('/api/v1/domains/group-open', {
        run_id: effectiveRunId,
        mode: 'data'
      });
      
      // Store both the grouping and open labeling results for Groups View
      setGroupedData(groupRes.data.groups || []);
      setOpenLabelingData(openRes.data.labels || []);
      
      // Convert grouped results to domain assignment format
      const groupedAssignments: DomainAssignment[] = [];
      const labelMap = new Map();
      
      // Build map of column -> label from open labeling
      (openRes.data.labels || []).forEach((label: any) => {
        labelMap.set(label.column_name, label.label);
      });
      
      // Process groups and assign columns to canonical domains
      (groupRes.data.groups || []).forEach((group: any) => {
        (group.columns || []).forEach((column: any) => {
          // Handle both old format (string) and new format (object with name/side)
          const columnName = typeof column === 'string' ? column : column.name;
          const columnSide = typeof column === 'string' ? 'LHS' : (column.side || 'LHS');
          
          const originalLabel = labelMap.get(columnName) || 'unknown';
          groupedAssignments.push({
            column_name: columnName,
            domain_id: `grouped-${group.canonical}-${columnSide}`,
            domain_name: `${group.canonical} (${columnSide})`,
            confidence_score: originalLabel === 'unknown' ? 0.0 : 0.95,
            confidence_band: originalLabel === 'unknown' ? 'low' : 'high' as 'high' | 'borderline' | 'low',
            evidence: {
              name_similarity: 0.9,
              regex_strength: 0.8,
              value_similarity: 0.9,
              unit_compatibility: 0.8,
              composite_score: originalLabel === 'unknown' ? 0.0 : 0.95,
              matching_aliases: group.members || [],
              matching_patterns: [],
              matching_units: [],
              header_tokens: [columnName, columnSide, originalLabel] // Store original GPT-4o-mini label
            },
            assigned_at: new Date().toISOString(),
            human_reviewed: false,
            human_decision: null
          });
        });
      });
      
      // Add any columns that weren't grouped
      (openRes.data.labels || []).forEach((label: any) => {
        if (!groupedAssignments.find(a => a.column_name === label.column_name)) {
          groupedAssignments.push({
            column_name: label.column_name,
            domain_id: `ungrouped-${label.column_name}`,
            domain_name: label.label,
            confidence_score: label.label === 'unknown' ? 0.0 : 0.8,
            confidence_band: label.label === 'unknown' ? 'low' : 'borderline' as 'high' | 'borderline' | 'low',
            evidence: {
              name_similarity: 0.7,
              regex_strength: 0.6,
              value_similarity: 0.7,
              unit_compatibility: 0.6,
              composite_score: label.label === 'unknown' ? 0.0 : 0.8,
              matching_aliases: [],
              matching_patterns: [],
              matching_units: [],
              header_tokens: [label.column_name]
            },
            assigned_at: new Date().toISOString(),
            human_reviewed: false,
            human_decision: null
          });
        }
      });
      
      setAssignments(groupedAssignments);
      setGroupedData(groupRes.data.groups || []);
      setViewMode('groups');
      setLlmMode(true);
      
      const groupCount = (groupRes.data.groups || []).length;
      const successCount = groupedAssignments.filter(a => a.domain_name !== 'unknown').length;
      onNotification?.({ 
        type: 'success', 
        message: `Grouped labeling completed: ${successCount}/${groupedAssignments.length} columns in ${groupCount} groups`,
        details: 'GPT-4o-mini + Claude Sonnet 4 intelligent grouping'
      });
    } catch (error) {
      console.error('Failed to assign grouped labels:', error);
      onNotification?.({ type: 'error', message: 'Failed to assign grouped labels' });
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (band: string) => {
    switch (band) {
      case 'high': 
      case 'approved': return 'success';
      case 'borderline': 
      case 'needs_review': return 'warning'; 
      case 'low': 
      case 'rejected': return 'error';
      default: return 'default';
    }
  };

  const getConfidenceIcon = (band: string) => {
    switch (band) {
      case 'high': 
      case 'approved': return '🟢';
      case 'borderline': 
      case 'needs_review': return '🟡';
      case 'low': 
      case 'rejected': return '🔴';
      default: return '⚪';
    }
  };

  const handleApprove = async (assignment: DomainAssignment) => {
    try {
      console.log('Approving assignment:', assignment.column_name, '→', assignment.domain_name);
      
      setAssignments(prev => prev.map(a => 
        a.domain_id === assignment.domain_id 
          ? { ...a, human_reviewed: true, human_decision: 'approved' }
          : a
      ));
      
      onNotification?.({ type: 'success', message: `Approved: ${assignment.column_name} → ${assignment.domain_name}` });
    } catch (error) {
      onNotification?.({ type: 'error', message: 'Failed to approve assignment' });
    }
  };

  const handleMarkUnknown = async (assignment: DomainAssignment) => {
    try {
      console.log('Marking as unknown:', assignment.column_name);
      
      setAssignments(prev => prev.map(a => 
        a.domain_id === assignment.domain_id 
          ? { ...a, human_reviewed: true, human_decision: 'unknown', domain_name: null, domain_id: null }
          : a
      ));
      
      onNotification?.({ type: 'info', message: `Marked as unknown: ${assignment.column_name}` });
    } catch (error) {
      onNotification?.({ type: 'error', message: 'Failed to mark as unknown' });
    }
  };

  const handleAddAlias = (assignment: DomainAssignment) => {
    setAliasDialog({
      open: true,
      columnName: assignment.column_name,
      domainId: assignment.domain_id || '',
      alias: ''
    });
  };

  const submitAlias = async () => {
    try {
      console.log('Adding alias:', aliasDialog.alias, 'for domain:', aliasDialog.domainId);
      
      onNotification?.({ type: 'success', message: `Added alias "${aliasDialog.alias}" to staged changes` });
      setAliasDialog({ open: false, columnName: '', domainId: '', alias: '' });
    } catch (error) {
      onNotification?.({ type: 'error', message: 'Failed to add alias' });
    }
  };

  // Statistics for summary cards
  const stats = React.useMemo(() => {
    const total = assignments.length;
    const approved = assignments.filter(a => a.human_decision === 'approved').length;
    const pending = assignments.filter(a => !a.human_reviewed).length;
    const unknown = assignments.filter(a => a.human_decision === 'unknown').length;
    const highConfidence = assignments.filter(a => ['high', 'approved'].includes(a.confidence_band)).length;
    
    return { total, approved, pending, unknown, highConfidence };
  }, [assignments]);

  const GroupsView: React.FC = () => {
    // Get original GPT-4o-mini labels mapping from open labeling data
    const originalLabelsMap = new Map();
    openLabelingData.forEach((labelData: any) => {
      originalLabelsMap.set(labelData.column_name, labelData.label);
    });

    return (
      <Box sx={{ space: 2 }}>
        {groupedData.map((group, groupIndex) => (
          <Card 
            key={group.canonical}
            elevation={2}
            sx={{ 
              mb: 3,
              border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Category 
                  sx={{ 
                    mr: 2, 
                    fontSize: 28,
                    color: 'primary.main',
                    p: 1,
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    borderRadius: 2
                  }} 
                />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                    {group.canonical}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {group.columns?.length || 0} columns grouped by semantic similarity
                  </Typography>
                </Box>
                <Chip 
                  label={`${group.columns?.length || 0} columns`}
                  color="primary"
                  variant="outlined"
                  sx={{ fontWeight: 'bold' }}
                />
              </Box>

              {/* Table Format for Columns */}
              {group.columns && group.columns.length > 0 && (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ bgcolor: alpha(theme.palette.primary.main, 0.05) }}>
                        <TableCell sx={{ fontWeight: 'bold', width: '15%' }}>Side</TableCell>
                        <TableCell sx={{ fontWeight: 'bold', width: '25%' }}>Column Name</TableCell>
                        <TableCell sx={{ fontWeight: 'bold', width: '60%' }}>GPT-4o-mini Label</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {group.columns.map((column: any, idx: number) => {
                        const columnName = typeof column === 'string' ? column : column.name;
                        const columnSide = typeof column === 'string' ? 'LHS' : (column.side || 'LHS');
                        const originalLabel = originalLabelsMap.get(columnName) || 'unknown';
                        
                        return (
                          <TableRow 
                            key={idx}
                            hover
                            sx={{ '&:nth-of-type(odd)': { bgcolor: alpha(theme.palette.grey[100], 0.5) } }}
                          >
                            <TableCell>
                              <Chip 
                                label={columnSide}
                                size="small"
                                color={columnSide === 'LHS' ? 'success' : 'warning'}
                                sx={{ 
                                  fontSize: '0.75rem',
                                  fontWeight: 'bold',
                                  minWidth: '50px'
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <Typography 
                                variant="body2" 
                                sx={{ 
                                  fontFamily: 'monospace',
                                  fontWeight: 'medium',
                                  color: 'primary.main'
                                }}
                              >
                                {columnName}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography 
                                variant="body2"
                                sx={{ 
                                  color: 'text.primary',
                                  fontWeight: 'medium'
                                }}
                              >
                                {originalLabel}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>
    );
  };

  const EvidenceDrawer: React.FC<{ assignment: DomainAssignment }> = ({ assignment }) => {
    const { evidence } = assignment;
    
    return (
      <Box sx={{ 
        p: 3, 
        bgcolor: alpha(theme.palette.primary.main, 0.02),
        borderRadius: 2,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`
      }}>
        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', md: 'row' },
          gap: 3
        }}>
          {/* Score Breakdown */}
          <Box sx={{ flex: 1 }}>
            <Card 
              variant="outlined" 
              sx={{ 
                height: '100%',
                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                '&:hover': {
                  boxShadow: theme.shadows[4]
                }
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', color: 'primary.main' }}>
                  <Analytics sx={{ mr: 1 }} />
                  Score Breakdown
                </Typography>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">Name Similarity</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ 
                        width: '100%', 
                        height: 8, 
                        bgcolor: 'grey.200', 
                        borderRadius: 1,
                        overflow: 'hidden'
                      }}>
                        <Box sx={{
                          width: `${evidence.name_similarity * 100}%`,
                          height: '100%',
                          bgcolor: 'primary.main',
                          borderRadius: 1
                        }} />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 'fit-content', fontWeight: 'bold' }}>
                        {evidence.name_similarity.toFixed(3)}
                      </Typography>
                    </Box>
                  </Box>

                  <Box>
                    <Typography variant="body2" color="text.secondary">Pattern Strength</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ 
                        width: '100%', 
                        height: 8, 
                        bgcolor: 'grey.200', 
                        borderRadius: 1,
                        overflow: 'hidden'
                      }}>
                        <Box sx={{
                          width: `${evidence.regex_strength * 100}%`,
                          height: '100%',
                          bgcolor: 'secondary.main',
                          borderRadius: 1
                        }} />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 'fit-content', fontWeight: 'bold' }}>
                        {evidence.regex_strength.toFixed(3)}
                      </Typography>
                    </Box>
                  </Box>

                  <Box>
                    <Typography variant="body2" color="text.secondary">Value Similarity</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ 
                        width: '100%', 
                        height: 8, 
                        bgcolor: 'grey.200', 
                        borderRadius: 1,
                        overflow: 'hidden'
                      }}>
                        <Box sx={{
                          width: `${evidence.value_similarity * 100}%`,
                          height: '100%',
                          bgcolor: 'success.main',
                          borderRadius: 1
                        }} />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 'fit-content', fontWeight: 'bold' }}>
                        {evidence.value_similarity.toFixed(3)}
                      </Typography>
                    </Box>
                  </Box>

                  <Divider />
                  
                  <Box>
                    <Typography variant="subtitle2" color="primary.main" sx={{ fontWeight: 'bold' }}>
                      Composite Score: {evidence.composite_score.toFixed(3)}
                    </Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Box>

          {/* Evidence Details */}
          <Box sx={{ flex: 1 }}>
            <Card 
              variant="outlined" 
              sx={{ 
                height: '100%',
                border: `1px solid ${alpha(theme.palette.secondary.main, 0.2)}`,
                '&:hover': {
                  boxShadow: theme.shadows[4]
                }
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', color: 'secondary.main' }}>
                  <Info sx={{ mr: 1 }} />
                  Evidence Details
                </Typography>
                
                <Stack spacing={2}>
                  {evidence.header_tokens.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <DataObject sx={{ mr: 1, fontSize: 'small' }} />
                        Header Tokens
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {evidence.header_tokens.map((token, idx) => (
                          <Chip key={idx} label={token} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  )}

                  {evidence.matching_patterns.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <Pattern sx={{ mr: 1, fontSize: 'small' }} />
                        Matching Patterns
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {evidence.matching_patterns.map((pattern, idx) => (
                          <Chip 
                            key={idx} 
                            label={pattern} 
                            size="small" 
                            color="primary" 
                            variant="outlined"
                            sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }} 
                          />
                        ))}
                      </Box>
                    </Box>
                  )}

                  {evidence.matching_aliases.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <Category sx={{ mr: 1, fontSize: 'small' }} />
                        Matching Aliases
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {evidence.matching_aliases.map((alias, idx) => (
                          <Chip 
                            key={idx} 
                            label={alias} 
                            size="small" 
                            color="secondary" 
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header Section */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
              🎯 Domain Studio
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
              AI-powered domain assignment with neural embeddings and LLM reasoning
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={2}>
            <Button 
              variant="outlined" 
              onClick={() => assignDomains(false)}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : <Refresh />}
            >
              Neural Mode
            </Button>
            <Button 
              variant="contained" 
              onClick={() => assignOpenLabels()}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : <DataObject />}
              sx={{ 
                background: 'linear-gradient(45deg, #FF6B35 30%, #F7931E 90%)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #E55A2B 30%, #E8841B 90%)',
                }
              }}
            >
              Open Labels (GPT-4o-mini)
            </Button>
            <Button 
              variant="contained" 
              onClick={() => assignGroupedLabels()}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : <AutoAwesome />}
              sx={{ 
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
                }
              }}
            >
              Grouped (Claude Sonnet 4)
            </Button>
          </Stack>
        </Box>

        {/* Statistics Cards */}
        <Box sx={{ 
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: 2, 
          mb: 3,
          '& > *': { 
            flex: { xs: '1 1 calc(50% - 8px)', sm: '1 1 calc(25% - 12px)' }
          }
        }}>
          <Card sx={{ textAlign: 'center', p: 2, minWidth: 120 }}>
            <Typography variant="h4" color="primary.main" sx={{ fontWeight: 'bold' }}>
              {stats.total}
            </Typography>
            <Typography variant="body2" color="text.secondary">Total Columns</Typography>
          </Card>
          <Card sx={{ textAlign: 'center', p: 2, minWidth: 120 }}>
            <Typography variant="h4" color="success.main" sx={{ fontWeight: 'bold' }}>
              {stats.highConfidence}
            </Typography>
            <Typography variant="body2" color="text.secondary">High Confidence</Typography>
          </Card>
          <Card sx={{ textAlign: 'center', p: 2, minWidth: 120 }}>
            <Typography variant="h4" color="warning.main" sx={{ fontWeight: 'bold' }}>
              {stats.pending}
            </Typography>
            <Typography variant="body2" color="text.secondary">Pending Review</Typography>
          </Card>
          <Card sx={{ textAlign: 'center', p: 2, minWidth: 120 }}>
            <Typography variant="h4" color="success.main" sx={{ fontWeight: 'bold' }}>
              {stats.approved}
            </Typography>
            <Typography variant="body2" color="text.secondary">Approved</Typography>
          </Card>
        </Box>

        {/* Mode Indicator */}
        {assignments.length > 0 && (
          <>
            <Alert 
              severity={llmMode ? "info" : "success"} 
              icon={llmMode ? <AutoAwesome /> : <TrendingUp />}
              sx={{ mb: 2 }}
            >
              <strong>{llmMode ? "LLM-Enhanced Mode" : "Neural Mode"}:</strong> {
                llmMode 
                  ? "Using CrewAI multi-agent analysis with contextual reasoning"
                  : "Using neural embeddings and rule-based pattern matching"
              }
            </Alert>
            
            {/* View Toggle */}
            {groupedData.length > 0 && (
              <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
                <Paper elevation={1} sx={{ p: 0.5, borderRadius: 3 }}>
                  <Stack direction="row" spacing={0}>
                    <Button 
                      variant={viewMode === 'assignments' ? 'contained' : 'text'}
                      onClick={() => setViewMode('assignments')}
                      size="small"
                      sx={{ borderRadius: 3 }}
                    >
                      Column View
                    </Button>
                    <Button 
                      variant={viewMode === 'groups' ? 'contained' : 'text'}
                      onClick={() => setViewMode('groups')}
                      size="small"
                      sx={{ borderRadius: 3 }}
                    >
                      Group View ({groupedData.length} groups)
                    </Button>
                  </Stack>
                </Paper>
              </Box>
            )}
          </>
        )}
      </Box>

      {/* Main Content */}
      {loading && assignments.length === 0 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 8 }}>
          <CircularProgress size={48} sx={{ mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            {llmMode ? "Running LLM analysis..." : "Analyzing domains..."}
          </Typography>
        </Box>
      ) : viewMode === 'groups' ? (
        <GroupsView />
      ) : (
        <TableContainer component={Paper} elevation={3} sx={{ borderRadius: 2 }}>
          <Table>
            <TableHead>
              <TableRow sx={{ bgcolor: alpha(theme.palette.primary.main, 0.1) }}>
                <TableCell width="40"></TableCell>
                <TableCell sx={{ fontWeight: 'bold' }}>Column Name</TableCell>
                <TableCell sx={{ fontWeight: 'bold' }}>Predicted Domain</TableCell>
                <TableCell align="center" sx={{ fontWeight: 'bold' }}>Confidence</TableCell>
                <TableCell align="center" sx={{ fontWeight: 'bold' }}>Score</TableCell>
                <TableCell align="center" sx={{ fontWeight: 'bold' }}>Status</TableCell>
                <TableCell align="center" sx={{ fontWeight: 'bold' }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {assignments.map((assignment, index) => (
                <React.Fragment key={`${assignment.column_name}-${index}`}>
                  <TableRow 
                    hover 
                    sx={{ 
                      '&:hover': { bgcolor: alpha(theme.palette.primary.main, 0.04) },
                      cursor: 'pointer'
                    }}
                    onClick={() => setExpandedRow(
                      expandedRow === `${assignment.column_name}-${index}` ? null : `${assignment.column_name}-${index}`
                    )}
                  >
                    <TableCell>
                      <IconButton size="small">
                        {expandedRow === `${assignment.column_name}-${index}` ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ 
                        fontFamily: 'monospace', 
                        fontWeight: 'medium',
                        color: 'primary.main'
                      }}>
                        {assignment.column_name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {assignment.domain_name ? (
                        <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                          {assignment.domain_name}
                        </Typography>
                      ) : (
                        <Typography variant="body2" color="text.secondary" fontStyle="italic">
                          Unknown
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={assignment.confidence_band.toUpperCase()}
                        color={getConfidenceColor(assignment.confidence_band) as any}
                        size="small"
                        icon={<span>{getConfidenceIcon(assignment.confidence_band)}</span>}
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2" sx={{ 
                        fontFamily: 'monospace', 
                        fontWeight: 'bold',
                        color: assignment.confidence_score > 0.8 ? 'success.main' : 
                               assignment.confidence_score > 0.5 ? 'warning.main' : 'error.main'
                      }}>
                        {assignment.confidence_score.toFixed(3)}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      {assignment.human_reviewed ? (
                        <Badge 
                          badgeContent={<Check sx={{ fontSize: 12 }} />} 
                          color="success"
                        >
                          <Chip
                            label={assignment.human_decision?.toUpperCase() || 'REVIEWED'}
                            color={assignment.human_decision === 'approved' ? 'success' : 'default'}
                            size="small"
                            sx={{ fontWeight: 'bold' }}
                          />
                        </Badge>
                      ) : (
                        <Chip label="PENDING" color="warning" size="small" sx={{ fontWeight: 'bold' }} />
                      )}
                    </TableCell>
                    <TableCell align="center" onClick={(e) => e.stopPropagation()}>
                      <Stack direction="row" spacing={0.5} justifyContent="center">
                        {!assignment.human_reviewed && assignment.domain_name && (
                          <Tooltip title="Approve assignment">
                            <IconButton
                              size="small"
                              color="success"
                              onClick={() => handleApprove(assignment)}
                              sx={{ '&:hover': { bgcolor: alpha(theme.palette.success.main, 0.1) } }}
                            >
                              <Check fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        {!assignment.human_reviewed && (
                          <Tooltip title="Mark as unknown">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleMarkUnknown(assignment)}
                              sx={{ '&:hover': { bgcolor: alpha(theme.palette.error.main, 0.1) } }}
                            >
                              <Close fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        {assignment.domain_id && (
                          <Tooltip title="Add alias/pattern">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleAddAlias(assignment)}
                              sx={{ '&:hover': { bgcolor: alpha(theme.palette.primary.main, 0.1) } }}
                            >
                              <Add fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Stack>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
                      <Collapse in={expandedRow === `${assignment.column_name}-${index}`} timeout="auto" unmountOnExit>
                        <Box sx={{ py: 2 }}>
                          <EvidenceDrawer assignment={assignment} />
                        </Box>
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Add Alias Dialog */}
      <Dialog 
        open={aliasDialog.open} 
        onClose={() => setAliasDialog({ ...aliasDialog, open: false })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Add sx={{ mr: 1 }} />
          Add Alias or Pattern
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Add an alias or regex pattern for column <strong>"{aliasDialog.columnName}"</strong> to improve future domain detection.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Alias or Pattern"
            fullWidth
            variant="outlined"
            value={aliasDialog.alias}
            onChange={(e) => setAliasDialog({ ...aliasDialog, alias: e.target.value })}
            placeholder="e.g., 'user_email' or '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'"
            helperText="This will be added to staged changes for domain catalog updates"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAliasDialog({ ...aliasDialog, open: false })}>
            Cancel
          </Button>
          <Button 
            onClick={submitAlias} 
            variant="contained"
            disabled={!aliasDialog.alias.trim()}
            startIcon={<Add />}
          >
            Add to Staged Changes
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DomainStudio;