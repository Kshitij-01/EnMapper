import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Box,
  Typography,
  Alert,
  Snackbar,
  LinearProgress,
  AppBar,
  Toolbar,
  Tabs,
  Tab,
  Fade,
  alpha,
} from '@mui/material';
import { StartRunScreen } from './components/StartRunScreen';
import DomainStudio from './components/DomainStudio';
import './App.css';

// Create beautiful modern theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563eb',
      light: '#60a5fa',
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#7c3aed',
      light: '#a78bfa',
      dark: '#5b21b6',
    },
    background: {
      default: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      paper: 'rgba(255, 255, 255, 0.98)',
    },
    text: {
      primary: '#1a202c',
      secondary: '#4a5568',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      fontSize: '3.5rem',
      background: 'linear-gradient(45deg, #2563eb, #7c3aed)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
    },
    h4: {
      fontWeight: 700,
      color: '#1e293b',
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(255, 255, 255, 0.98)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
          color: '#1a202c',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 12,
          padding: '12px 24px',
        },
        contained: {
          background: 'linear-gradient(45deg, #2563eb 30%, #7c3aed 90%)',
          boxShadow: '0 4px 20px 0 rgba(37, 99, 235, 0.3)',
          '&:hover': {
            background: 'linear-gradient(45deg, #1d4ed8 30%, #5b21b6 90%)',
            boxShadow: '0 6px 25px 0 rgba(37, 99, 235, 0.4)',
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '1rem',
          color: 'rgba(255, 255, 255, 0.8)',
          '&.Mui-selected': {
            color: '#ffffff',
            fontWeight: 700,
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
});

export interface AppNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  details?: string;
}

export interface AppState {
  loading: boolean;
  notifications: AppNotification[];
}

function App() {
  const [currentTab, setCurrentTab] = useState(1); // Start with Domain Studio
  const [appState, setAppState] = useState<AppState>({
    loading: false,
    notifications: []
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const addNotification = (notification: Omit<AppNotification, 'id'>) => {
    const newNotification: AppNotification = {
      ...notification,
      id: Date.now().toString()
    };
    
    setAppState(prev => ({
      ...prev,
      notifications: [...prev.notifications, newNotification]
    }));

    // Auto-remove after 5 seconds
    setTimeout(() => {
      setAppState(prev => ({
        ...prev,
        notifications: prev.notifications.filter(n => n.id !== newNotification.id)
      }));
    }, 5000);
  };

  const removeNotification = (id: string) => {
    setAppState(prev => ({
      ...prev,
      notifications: prev.notifications.filter(n => n.id !== id)
    }));
  };

  const setLoading = (loading: boolean) => {
    setAppState(prev => ({ ...prev, loading }));
  };

  const handleError = (error: any) => {
    console.error('Application error:', error);
    addNotification({
      type: 'error',
      message: 'An error occurred',
      details: error.message || 'Unknown error'
    });
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.15) 0%, transparent 50%)',
          pointerEvents: 'none',
        }
      }}>
        <AppBar position="static" elevation={0}>
          <Toolbar sx={{ minHeight: '80px !important' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              <Typography 
                variant="h4" 
                component="div" 
                sx={{ 
                  fontWeight: 800,
                  color: 'white',
                  fontSize: '2rem',
                  textShadow: '0 2px 4px rgba(0,0,0,0.3)'
                }}
              >
                🎯 EnMapper
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  ml: 2, 
                  color: '#000000 !important',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  textShadow: '1px 1px 2px rgba(255,255,255,0.8)'
                }}
              >
                AI-Powered Data Mapping Platform
              </Typography>
            </Box>
            <Tabs 
              value={currentTab} 
              onChange={handleTabChange} 
              sx={{ 
                '& .MuiTabs-indicator': { 
                  backgroundColor: '#ffffff',
                  height: 3,
                  borderRadius: '3px 3px 0 0'
                }
              }}
            >
              <Tab label="🚀 Start Run" />
              <Tab label="🎨 Domain Studio" />
            </Tabs>
          </Toolbar>
        </AppBar>

        {appState.loading && (
          <LinearProgress 
            sx={{ 
              height: 3,
              '& .MuiLinearProgress-bar': {
                background: 'linear-gradient(45deg, #ffffff 30%, rgba(255,255,255,0.8) 90%)'
              }
            }} 
          />
        )}

        <Container maxWidth="xl" sx={{ py: 4, position: 'relative', zIndex: 1 }}>
          <Fade in={true} timeout={800}>
            <Box>
              {currentTab === 0 && (
                <Box sx={{ 
                  background: 'rgba(255, 255, 255, 0.98)',
                  backdropFilter: 'blur(20px)',
                  borderRadius: 4,
                  p: 4,
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
                  color: '#1a202c',
                }}>
                  <StartRunScreen 
                    onNotification={addNotification}
                    onLoading={setLoading}
                    onError={handleError}
                    isLoading={appState.loading}
                  />
                </Box>
              )}

              {currentTab === 1 && (
                <Box sx={{ 
                  background: 'rgba(255, 255, 255, 0.98)',
                  backdropFilter: 'blur(20px)',
                  borderRadius: 4,
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
                  overflow: 'hidden',
                  color: '#1a202c',
                }}>
                  <DomainStudio
                    onNotification={addNotification}
                  />
                </Box>
              )}
            </Box>
          </Fade>
        </Container>

        {/* Notifications */}
        {appState.notifications.map((notification) => (
          <Snackbar
            key={notification.id}
            open={true}
            autoHideDuration={5000}
            onClose={() => removeNotification(notification.id)}
            anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
            sx={{ mt: 10 }}
          >
            <Alert 
              onClose={() => removeNotification(notification.id)} 
              severity={notification.type}
              variant="filled"
              sx={{ 
                minWidth: '300px',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.15)'
              }}
            >
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {notification.message}
              </Typography>
              {notification.details && (
                <Typography variant="caption" sx={{ opacity: 0.9 }}>
                  {notification.details}
                </Typography>
              )}
            </Alert>
          </Snackbar>
        ))}
      </Box>
    </ThemeProvider>
  );
}

export default App;