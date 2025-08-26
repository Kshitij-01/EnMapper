import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🔵 API Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
    return config;
  },
  (error) => {
    console.error('🔴 API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging and error handling
api.interceptors.response.use(
  (response) => {
    console.log(`🟢 API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    console.error('🔴 API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Types
export interface DataSource {
  type: 'file' | 'sql';
  location: string;
  format?: string;
  connection_params?: Record<string, any>;
}

export interface CreateRunRequest {
  data_source: DataSource;
  mode: 'metadata_only' | 'data_mode';
  lane_hint?: 'interactive' | 'flex' | 'batch';
  pii_masking_enabled: boolean;
  budget_caps?: {
    tokens: number;
    usd: number;
    wall_time_s: number;
  };
  run_name?: string;
  description?: string;
  tags?: string[];
}

export interface CreateRunResponse {
  success: boolean;
  run_id: string;
  status: string;
  contract_hash: string;
  estimated_cost?: {
    tokens: number;
    usd: number;
    confidence: string;
  };
  created_at: string;
  message: string;
}

export interface RunInfo {
  run_id: string;
  status: string;
  mode: string;
  lane_hint?: string;
  created_at: string;
  updated_at: string;
  run_name?: string;
  description?: string;
  tags: string[];
  current_stage?: string;
  completion_percentage: number;
  tokens_used: number;
  cost_usd: number;
  wall_time_s: number;
}

export interface GetRunResponse {
  run: RunInfo;
  contract: Record<string, any>;
  artifacts: Record<string, any>;
  ledger_events: Array<Record<string, any>>;
}

export interface DatabaseConnectionTest {
  connection_type: string;
  connection_params: Record<string, any>;
}

export interface DatabaseConnectionTestResponse {
  success: boolean;
  database_type: string;
  connection_info: Record<string, any>;
  response_time_ms?: number;
  server_version?: string;
  schema_info?: Record<string, any>;
  error_message?: string;
}

export interface CostEstimateRequest {
  data_source: DataSource;
  mode: 'metadata_only' | 'data_mode';
  sample_size?: number;
}

export interface CostEstimateResponse {
  estimated_tokens: number;
  estimated_cost_usd: number;
  processing_time_estimate_s: number;
  breakdown: Record<string, any>;
  confidence: string;
}

export interface PolicyCheckRequest {
  run_contract: Record<string, any>;
}

export interface PolicyCheckResponse {
  allowed: boolean;
  violations: string[];
  warnings: string[];
  recommendations: string[];
}

// API Service Class
export class EnMapperAPI {
  // Health Check
  static async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await api.get('/health');
    return response.data;
  }

  // Run Management
  static async createRun(request: CreateRunRequest): Promise<CreateRunResponse> {
    const response = await api.post('/api/v1/runs', request);
    return response.data;
  }

  static async getRun(runId: string): Promise<GetRunResponse> {
    const response = await api.get(`/api/v1/runs/${runId}`);
    return response.data;
  }

  // Database Testing
  static async testDatabaseConnection(request: DatabaseConnectionTest): Promise<DatabaseConnectionTestResponse> {
    const response = await api.post('/api/v1/database/test', request);
    return response.data;
  }

  // Cost Estimation
  static async estimateCost(request: CostEstimateRequest): Promise<CostEstimateResponse> {
    const response = await api.post('/api/v1/cost/estimate', request);
    return response.data;
  }

  // Policy Validation
  static async checkPolicy(request: PolicyCheckRequest): Promise<PolicyCheckResponse> {
    const response = await api.post('/api/v1/policy/check', request);
    return response.data;
  }

  // File Upload (multipart)
  static async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/v1/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  // Service Status
  static async getServiceStatus(): Promise<any> {
    const response = await api.get('/status');
    return response.data;
  }

  // API Info
  static async getAPIInfo(): Promise<any> {
    const response = await api.get('/info');
    return response.data;
  }
}

// React Hooks for API calls
export const useAPI = () => {
  return {
    healthCheck: EnMapperAPI.healthCheck,
    createRun: EnMapperAPI.createRun,
    getRun: EnMapperAPI.getRun,
    testDatabaseConnection: EnMapperAPI.testDatabaseConnection,
    estimateCost: EnMapperAPI.estimateCost,
    checkPolicy: EnMapperAPI.checkPolicy,
    uploadFile: EnMapperAPI.uploadFile,
    getServiceStatus: EnMapperAPI.getServiceStatus,
    getAPIInfo: EnMapperAPI.getAPIInfo,
  };
};

export default api;
