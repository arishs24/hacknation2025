import {
  HealthSchema,
  TargetSchema,
  PredictRequestSchema,
  PredictResponseSchema,
  BatchRequestSchema,
  BatchResponseSchema,
  ExplainRequestSchema,
  ExplainResponseSchema,
  MetricsResponseSchema,
  type Health,
  type Target,
  type PredictRequest,
  type PredictResponse,
  type BatchRequest,
  type BatchResponse,
  type ExplainRequest,
  type ExplainResponse,
  type MetricsResponse,
} from './schemas'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
const TIMEOUT_MS = 20000

class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: Response
  ) {
    super(message)
    this.name = 'APIError'
  }
}

async function fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })
    
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError('Request timeout after 20 seconds')
    }
    throw error
  }
}

async function retryFetch(url: string, options: RequestInit = {}, maxRetries = 3): Promise<Response> {
  let lastError: Error

  for (let i = 0; i <= maxRetries; i++) {
    try {
      const response = await fetchWithTimeout(url, options)
      
      if (!response.ok) {
        const errorText = await response.text()
        throw new APIError(
          `HTTP ${response.status}: ${errorText}`,
          response.status,
          response
        )
      }
      
      return response
    } catch (error) {
      lastError = error as Error
      
      // Don't retry POST requests by default
      if (options.method === 'POST' && i === 0) {
        break
      }
      
      // Don't retry on 4xx errors (client errors)
      if (error instanceof APIError && error.status && error.status >= 400 && error.status < 500) {
        break
      }
      
      if (i < maxRetries) {
        // Exponential backoff: 1s, 2s, 4s
        const delay = Math.pow(2, i) * 1000
        await new Promise(resolve => setTimeout(resolve, delay))
      }
    }
  }

  throw lastError!
}

export const api = {
  async health(): Promise<Health> {
    const response = await retryFetch(`${API_BASE}/health`)
    const data = await response.json()
    return HealthSchema.parse(data)
  },

  async targets(): Promise<Target[]> {
    const response = await retryFetch(`${API_BASE}/targets`)
    const data = await response.json()
    return data.map((item: unknown) => TargetSchema.parse(item))
  },

  async batch(compounds: any[], options: {
    target_id?: string
    calibrate?: boolean
    standardize_salts?: boolean
    skip_invalid?: boolean
  } = {}): Promise<any> {
    const response = await retryFetch(`${API_BASE}/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        compounds,
        ...options
      })
    })
    return response.json()
  },

  async predict(request: PredictRequest): Promise<PredictResponse> {
    const validatedRequest = PredictRequestSchema.parse(request)
    const response = await fetchWithTimeout(`${API_BASE}/predict`, {
      method: 'POST',
      body: JSON.stringify(validatedRequest),
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      throw new APIError(`Prediction failed: ${errorText}`, response.status, response)
    }
    
    const data = await response.json()
    return PredictResponseSchema.parse(data)
  },

  async predictBatch(request: BatchRequest): Promise<BatchResponse> {
    const validatedRequest = BatchRequestSchema.parse(request)
    const response = await fetchWithTimeout(`${API_BASE}/predict-batch`, {
      method: 'POST',
      body: JSON.stringify(validatedRequest),
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      throw new APIError(`Batch prediction failed: ${errorText}`, response.status, response)
    }
    
    const data = await response.json()
    return BatchResponseSchema.parse(data)
  },

  async explain(request: ExplainRequest): Promise<ExplainResponse> {
    const validatedRequest = ExplainRequestSchema.parse(request)
    const response = await fetchWithTimeout(`${API_BASE}/explain`, {
      method: 'POST',
      body: JSON.stringify(validatedRequest),
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      throw new APIError(`Explanation failed: ${errorText}`, response.status, response)
    }
    
    const data = await response.json()
    return ExplainResponseSchema.parse(data)
  },

  async metrics(): Promise<MetricsResponse> {
    const response = await retryFetch(`${API_BASE}/metrics`)
    const data = await response.json()
    return MetricsResponseSchema.parse(data)
  },
}

// Utility functions
export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError
}

export function getErrorMessage(error: unknown): string {
  if (isAPIError(error)) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}

export function getErrorDetails(error: unknown): { 
  message: string
  status?: number
  stack?: string 
} {
  if (isAPIError(error)) {
    return {
      message: error.message,
      status: error.status,
      stack: error.stack,
    }
  }
  if (error instanceof Error) {
    return {
      message: error.message,
      stack: error.stack,
    }
  }
  return {
    message: 'An unknown error occurred',
  }
}
