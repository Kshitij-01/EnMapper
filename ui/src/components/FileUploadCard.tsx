import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Stack,
  Chip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  InsertDriveFile as FileIcon,
  Check as CheckIcon,
  Delete as DeleteIcon,
  Error as ErrorIcon,
  SmartToy as AgentIcon,
} from '@mui/icons-material';

interface FileUploadCardProps {
  selected: boolean;
  onSelect: () => void;
  onNotification?: (notification: { type: 'success' | 'error' | 'warning' | 'info'; message: string; details?: string }) => void;
  onLoading?: (loading: boolean, operation?: string, progress?: number) => void;
  onError?: (error: Error | string, details?: string) => void;
  onFileUploaded?: (result: any) => void;
  agentMode?: boolean;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
  error?: string;
}

const FileUploadCard: React.FC<FileUploadCardProps> = ({ 
  selected, 
  onSelect, 
  onNotification, 
  onLoading, 
  onError, 
  onFileUploaded,
  agentMode: propAgentMode
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [agentMode, setAgentMode] = useState(() => {
    return propAgentMode !== undefined ? propAgentMode : true;
  });

  // Update local state when prop changes
  useEffect(() => {
    if (propAgentMode !== undefined) {
      setAgentMode(propAgentMode);
    }
  }, [propAgentMode]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    onSelect(); // Select this card when files are dropped
    
    // Handle accepted files
    acceptedFiles.forEach((file) => {
      const fileId = Math.random().toString(36).substr(2, 9);
      const newFile: UploadedFile = {
        file,
        id: fileId,
        status: 'uploading',
        progress: 0,
      };
      
      setUploadedFiles(prev => [...prev, newFile]);
      
      // Start real upload
      uploadFile(fileId, file);
    });

    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      const fileId = Math.random().toString(36).substr(2, 9);
      const errorMessages = errors.map((e: any) => e.message).join(', ');
      const newFile: UploadedFile = {
        file,
        id: fileId,
        status: 'error',
        progress: 0,
        error: errorMessages,
      };
      
      setUploadedFiles(prev => [...prev, newFile]);
    });
  }, [onSelect]);

  const uploadFile = async (fileId: string, file: File) => {
    try {
      console.log('🚀 Starting file upload:', file.name);
      onLoading?.(true, `Uploading ${file.name}...`, 0);
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('run_name', `Upload: ${file.name}`);
      formData.append('description', `Uploaded file: ${file.name}`);
      formData.append('mode', 'data_mode');
      formData.append('pii_masking_enabled', 'true');
      formData.append('lane_hint', 'interactive');
      formData.append('agent_mode', agentMode.toString());
      
      console.log('📝 FormData prepared, agent_mode:', agentMode);

      // Simulate progress during upload
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 15;
        if (progress <= 90) {
          setUploadedFiles(prev =>
            prev.map(f =>
              f.id === fileId ? { ...f, progress } : f
            )
          );
          onLoading?.(true, `Uploading ${file.name}...`, progress);
        }
      }, 200);

      console.log('🌐 Making request to:', 'http://localhost:8000/api/v1/files/upload');
      
      const response = await fetch('http://localhost:8000/api/v1/files/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      console.log('📡 Response received:', response.status, response.statusText);

      if (!response.ok) {
        const errorData = await response.text();
        console.error('❌ Upload failed:', errorData);
        throw new Error(`Upload failed: ${errorData}`);
      }

      const result = await response.json();
      console.log('✅ Upload successful:', result);
      
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId ? { ...f, status: 'completed', progress: 100 } : f
        )
      );

      console.log('🎉 Calling onNotification with success');
      onNotification?.({
        type: 'success',
        message: `File uploaded successfully!`,
        details: `Run ID: ${result.run_id}`
      });

      console.log('📞 Calling onFileUploaded with result');
      onFileUploaded?.(result);
      onLoading?.(false);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      console.error('💥 Upload error caught:', error);
      
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId ? { ...f, status: 'error', error: errorMessage } : f
        )
      );

      console.log('🚨 Calling onError:', errorMessage);
      onError?.(errorMessage, `Failed to upload ${file.name}`);
      onLoading?.(false);
    }
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/json': ['.json'],
      'application/x-parquet': ['.parquet'],
      'text/plain': ['.txt'],
      'application/zip': ['.zip'],
      'application/x-zip-compressed': ['.zip'],
    },
    maxSize: 500 * 1024 * 1024, // 500MB
    multiple: true,
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const supportedFormats = ['CSV', 'TSV', 'JSON', 'Parquet', 'ZIP'];

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
            <UploadIcon
              sx={{
                fontSize: 48,
                color: selected ? 'primary.main' : 'text.secondary',
                mb: 1,
              }}
            />
            <Typography variant="h5" fontWeight={600}>
              Upload Files
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Drag and drop your data files here
            </Typography>
          </Box>

          {/* Dropzone */}
          <Box
            {...getRootProps()}
            sx={{
              border: 2,
              borderStyle: 'dashed',
              borderColor: isDragActive || dragActive ? 'primary.main' : 'divider',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              backgroundColor: isDragActive || dragActive ? 'rgba(37, 99, 235, 0.05)' : 'transparent',
              transition: 'all 0.2s ease-in-out',
              cursor: 'pointer',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'rgba(37, 99, 235, 0.02)',
              },
            }}
            className="upload-dropzone"
          >
            <input {...getInputProps()} />
            <Stack spacing={2} alignItems="center">
              <UploadIcon sx={{ fontSize: 40, color: 'text.secondary' }} />
              <Box>
                <Typography variant="body1" fontWeight={500}>
                  {isDragActive ? 'Drop files here' : 'Drag files here or click to browse'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Maximum file size: 500MB per file
                </Typography>
              </Box>
              <Button variant="outlined" size="small">
                Choose Files
              </Button>
            </Stack>
          </Box>

          {/* Agent Mode Toggle */}
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={agentMode}
                  onChange={(e) => setAgentMode(e.target.checked)}
                  color="primary"
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AgentIcon color={agentMode ? 'primary' : 'disabled'} />
                  <Typography variant="body2">
                    🧠 LLM Agent Mode (Code Generation & Execution)
                  </Typography>
                </Box>
              }
            />
            {agentMode && (
              <Alert severity="info" sx={{ mt: 1 }}>
                <Typography variant="caption">
                  🧠 LLM Agent will analyze your files, generate Python code dynamically, execute it with live terminal output, and standardize data for domain mapping.
                </Typography>
              </Alert>
            )}
          </Box>

          {/* Supported Formats */}
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Supported formats:
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {supportedFormats.map((format) => (
                <Chip
                  key={format}
                  label={format}
                  size="small"
                  variant="outlined"
                  color="primary"
                />
              ))}
            </Stack>
          </Box>

          {/* Uploaded Files */}
          {uploadedFiles.length > 0 && (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Uploaded Files ({uploadedFiles.length})
              </Typography>
              <List dense>
                {uploadedFiles.map((uploadedFile) => (
                  <ListItem
                    key={uploadedFile.id}
                    secondaryAction={
                      <IconButton
                        edge="end"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(uploadedFile.id);
                        }}
                        size="small"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    }
                    sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      mb: 1,
                      backgroundColor: 'background.paper',
                    }}
                  >
                    <ListItemIcon>
                      {uploadedFile.status === 'completed' ? (
                        <CheckIcon color="success" />
                      ) : uploadedFile.status === 'error' ? (
                        <ErrorIcon color="error" />
                      ) : (
                        <FileIcon color="primary" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={uploadedFile.file.name}
                      secondary={
                        <span>
                          <Typography variant="caption" color="text.secondary">
                            {formatFileSize(uploadedFile.file.size)}
                          </Typography>
                          {uploadedFile.status === 'uploading' && (
                            <LinearProgress
                              variant="determinate"
                              value={uploadedFile.progress}
                              sx={{ mt: 0.5, display: 'block' }}
                            />
                          )}
                          {uploadedFile.status === 'error' && (
                            <Alert severity="error" sx={{ mt: 0.5, display: 'block' }}>
                              <Typography variant="caption">
                                {uploadedFile.error}
                              </Typography>
                            </Alert>
                          )}
                        </span>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default FileUploadCard;
