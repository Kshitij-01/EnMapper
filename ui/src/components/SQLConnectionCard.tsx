import React, { useState } from 'react';
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
  Alert,
  Collapse,
  IconButton,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Storage as DatabaseIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Security as SecurityIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';

interface SQLConnectionCardProps {
  selected: boolean;
  onSelect: () => void;
}

interface ConnectionConfig {
  type: 'postgresql' | 'mysql' | 'sqlite';
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  ssl: boolean;
  readOnly: boolean;
  connectionTimeout: number;
}

interface ConnectionTestResult {
  success: boolean;
  message: string;
  details?: any;
}

const SQLConnectionCard: React.FC<SQLConnectionCardProps> = ({ selected, onSelect }) => {
  const [config, setConfig] = useState<ConnectionConfig>({
    type: 'postgresql',
    host: 'localhost',
    port: '5432',
    database: '',
    username: '',
    password: '',
    ssl: true,
    readOnly: true,
    connectionTimeout: 10,
  });
  
  const [showPassword, setShowPassword] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);

  const handleConfigChange = (field: keyof ConnectionConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
    setTestResult(null); // Clear test result when config changes
  };

  const handleDatabaseTypeChange = (type: 'postgresql' | 'mysql' | 'sqlite') => {
    const defaultPorts = {
      postgresql: '5432',
      mysql: '3306',
      sqlite: '',
    };
    
    setConfig(prev => ({
      ...prev,
      type,
      port: defaultPorts[type],
      host: type === 'sqlite' ? '' : prev.host,
    }));
    setTestResult(null);
  };

  const testConnection = async () => {
    setIsTestingConnection(true);
    onSelect(); // Select this card when testing connection
    
    try {
      // Simulate API call to test connection
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock successful result
      setTestResult({
        success: true,
        message: `Successfully connected to ${config.type} database`,
        details: {
          server_version: '14.5',
          schema_count: 3,
          table_count: 25,
        },
      });
    } catch (error) {
      setTestResult({
        success: false,
        message: 'Failed to connect to database',
      });
    } finally {
      setIsTestingConnection(false);
    }
  };

  const databaseTypes = [
    { value: 'postgresql', label: 'PostgreSQL', color: '#336791' },
    { value: 'mysql', label: 'MySQL', color: '#4479A1' },
    { value: 'sqlite', label: 'SQLite', color: '#003B57' },
  ];

  const isFormValid = () => {
    if (config.type === 'sqlite') {
      return config.database.length > 0;
    }
    return config.host && config.database && config.username;
  };

  return (
    <Card
      sx={{
        height: '100%',
        border: selected ? 2 : 1,
        borderColor: selected ? 'primary.main' : 'divider',
        backgroundColor: selected ? 'rgba(37, 99, 235, 0.02)' : 'background.paper',
        cursor: 'pointer',
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: 'primary.main',
          boxShadow: '0 4px 12px rgba(37, 99, 235, 0.15)',
        },
      }}
      onClick={onSelect}
    >
      <CardContent sx={{ p: 3 }}>
        <Stack spacing={3}>
          {/* Header */}
          <Box textAlign="center">
            <DatabaseIcon
              sx={{
                fontSize: 48,
                color: selected ? 'primary.main' : 'text.secondary',
                mb: 1,
              }}
            />
            <Typography variant="h5" fontWeight={600}>
              Connect SQL Database
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Connect to your existing database
            </Typography>
          </Box>

          {/* Database Type Selection */}
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Database Type
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {databaseTypes.map((type) => (
                <Chip
                  key={type.value}
                  label={type.label}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDatabaseTypeChange(type.value as any);
                  }}
                  color={config.type === type.value ? 'primary' : 'default'}
                  variant={config.type === type.value ? 'filled' : 'outlined'}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: type.color + '20',
                    },
                  }}
                />
              ))}
            </Stack>
          </Box>

          {/* Connection Form */}
          <Stack spacing={2}>
            {config.type !== 'sqlite' && (
              <>
                <Stack direction="row" spacing={2}>
                  <TextField
                    label="Host"
                    value={config.host}
                    onChange={(e) => handleConfigChange('host', e.target.value)}
                    fullWidth
                    size="small"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <TextField
                    label="Port"
                    value={config.port}
                    onChange={(e) => handleConfigChange('port', e.target.value)}
                    size="small"
                    sx={{ minWidth: 100 }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </Stack>
                
                <TextField
                  label="Username"
                  value={config.username}
                  onChange={(e) => handleConfigChange('username', e.target.value)}
                  fullWidth
                  size="small"
                  onClick={(e) => e.stopPropagation()}
                />
                
                <TextField
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  value={config.password}
                  onChange={(e) => handleConfigChange('password', e.target.value)}
                  fullWidth
                  size="small"
                  onClick={(e) => e.stopPropagation()}
                  InputProps={{
                    endAdornment: (
                      <IconButton
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowPassword(!showPassword);
                        }}
                        edge="end"
                        size="small"
                      >
                        {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                      </IconButton>
                    ),
                  }}
                />
              </>
            )}
            
            <TextField
              label={config.type === 'sqlite' ? 'Database File Path' : 'Database Name'}
              value={config.database}
              onChange={(e) => handleConfigChange('database', e.target.value)}
              fullWidth
              size="small"
              onClick={(e) => e.stopPropagation()}
              placeholder={config.type === 'sqlite' ? '/path/to/database.db' : 'database_name'}
            />
          </Stack>

          {/* Advanced Options */}
          <Box>
            <Button
              variant="text"
              size="small"
              startIcon={showAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              onClick={(e) => {
                e.stopPropagation();
                setShowAdvanced(!showAdvanced);
              }}
              sx={{ p: 0 }}
            >
              Advanced Options
            </Button>
            
            <Collapse in={showAdvanced}>
              <Stack spacing={2} sx={{ mt: 2 }}>
                {config.type !== 'sqlite' && (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.ssl}
                        onChange={(e) => handleConfigChange('ssl', e.target.checked)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    }
                    label="Use SSL"
                  />
                )}
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.readOnly}
                      onChange={(e) => handleConfigChange('readOnly', e.target.checked)}
                      onClick={(e) => e.stopPropagation()}
                    />
                  }
                  label="Read-only Access (Recommended)"
                />
                
                <TextField
                  label="Connection Timeout (seconds)"
                  type="number"
                  value={config.connectionTimeout}
                  onChange={(e) => handleConfigChange('connectionTimeout', parseInt(e.target.value))}
                  size="small"
                  onClick={(e) => e.stopPropagation()}
                  inputProps={{ min: 1, max: 300 }}
                />
              </Stack>
            </Collapse>
          </Box>

          {/* Security Notice */}
          <Alert
            severity="info"
            icon={<SecurityIcon />}
            sx={{ backgroundColor: 'rgba(33, 150, 243, 0.05)' }}
          >
            <Typography variant="body2">
              <strong>Security Note:</strong> We recommend using read-only database users 
              and SSL connections. Your credentials are not stored permanently.
            </Typography>
          </Alert>

          {/* Test Connection */}
          <Stack spacing={2}>
            <Button
              variant="outlined"
              fullWidth
              disabled={!isFormValid() || isTestingConnection}
              onClick={(e) => {
                e.stopPropagation();
                testConnection();
              }}
              startIcon={
                isTestingConnection ? (
                  <CircularProgress size={16} />
                ) : (
                  <DatabaseIcon />
                )
              }
            >
              {isTestingConnection ? 'Testing Connection...' : 'Test Connection'}
            </Button>

            {/* Test Result */}
            {testResult && (
              <Alert
                severity={testResult.success ? 'success' : 'error'}
                icon={testResult.success ? <CheckIcon /> : <ErrorIcon />}
              >
                <Typography variant="body2">
                  {testResult.message}
                  {testResult.success && testResult.details && (
                    <>
                      <br />
                      <strong>Server:</strong> {testResult.details.server_version} • 
                      <strong> Tables:</strong> {testResult.details.table_count}
                    </>
                  )}
                </Typography>
              </Alert>
            )}
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default SQLConnectionCard;
