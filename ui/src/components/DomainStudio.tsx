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
  List,
  ListItem,
  ListItemText,
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
  Psychology,
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
}

const DomainStudio: React.FC<DomainStudioProps> = ({ runId, onNotification }) => {
  const theme = useTheme();
  const [assignments, setAssignments] = useState<DomainAssignment[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [llmMode, setLlmMode] = useState(false);
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

  // Sample data for demo - in real app this would come from API
  const [sampleColumns] = useState([
    {
      name: "user_email_address",
      sample_values: ["john.doe@example.com", "jane@company.org", "admin@test.co.uk"],
      data_type: "varchar",
      null_count: 0,
      total_count: 200,
      unique_count: 198
    },
    {
      name: "contact_phone_num",
      sample_values: ["(555) 123-4567", "+1-555-987-6543", "555.111.2222"],
      data_type: "varchar",
      null_count: 3,
      total_count: 200,
      unique_count: 195
    },
    {
      name: "customer_first_name",
      sample_values: ["John", "Jane", "Michael", "Sarah"],
      data_type: "varchar",
      null_count: 1,
      total_count: 200,
      unique_count: 85
    },
    {
      name: "order_total_amount",
      sample_values: ["$1,234.56", "$999.99", "$0.99", "$10,000.00"],
      data_type: "decimal",
      null_count: 0,
      total_count: 200,
      unique_count: 200
    },
    {
      name: "created_timestamp",
      sample_values: ["2023-12-25T10:30:00Z", "2024-01-01T00:00:00Z", "2023-06-15T14:20:30Z"],
      data_type: "timestamp",
      null_count: 0,
      total_count: 200,
      unique_count: 185
    },
    {
      name: "mystery_field_xyz",
      sample_values: ["ABC123XYZ", "DEF456GHI", "JKL789MNO"],
      data_type: "varchar",
      null_count: 10,
      total_count: 200,
      unique_count: 200
    }
  ]);

  useEffect(() => {
    assignDomains();
  }, []);

  const assignDomains = async (useLlm = false) => {
    setLoading(true);
    try {
      const endpoint = useLlm ? '/api/v1/domains/assign-llm' : '/api/v1/domains/assign';
      const payload = {
        columns: sampleColumns,
        run_id: runId || 'domain_studio_demo',
        business_domain: 'e-commerce',
        table_name: 'customer_profiles',
        data_source: 'postgresql',
        budget_tier: 'balanced'
      };

      const response = await api.post(endpoint, payload);
      setAssignments(response.data.assignments);
      setLlmMode(useLlm);
      
      const message = useLlm 
        ? `LLM-enhanced domain assignments completed (${response.data.assignments.length} columns)`
        : `Domain assignments completed (${response.data.assignments.length} columns)`;
      
      onNotification?.({ type: 'success', message });
    } catch (error) {
      console.error('Failed to assign domains:', error);
      onNotification?.({ type: 'error', message: 'Failed to assign domains' });
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
        a.column_name === assignment.column_name 
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
        a.column_name === assignment.column_name 
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
              onClick={() => assignDomains(true)}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : <AutoAwesome />}
              sx={{ 
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
                }
              }}
            >
              LLM Enhanced
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
        )}
      </Box>

      {/* Main Table */}
      {loading && assignments.length === 0 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 8 }}>
          <CircularProgress size={48} sx={{ mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            {llmMode ? "Running LLM analysis..." : "Analyzing domains..."}
          </Typography>
        </Box>
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
              {assignments.map((assignment) => (
                <React.Fragment key={assignment.column_name}>
                  <TableRow 
                    hover 
                    sx={{ 
                      '&:hover': { bgcolor: alpha(theme.palette.primary.main, 0.04) },
                      cursor: 'pointer'
                    }}
                    onClick={() => setExpandedRow(
                      expandedRow === assignment.column_name ? null : assignment.column_name
                    )}
                  >
                    <TableCell>
                      <IconButton size="small">
                        {expandedRow === assignment.column_name ? <ExpandLess /> : <ExpandMore />}
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
                      <Collapse in={expandedRow === assignment.column_name} timeout="auto" unmountOnExit>
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