import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Target, PredictResponse, ExplainResponse, MetricsResponse } from './schemas'

interface PredictionState {
  // Current prediction inputs
  selectedTarget: Target | null
  smiles: string
  seed: number
  calibrate: boolean
  abstainBand: [number, number]
  
  // Current prediction results
  lastPrediction: PredictResponse | null
  lastExplanation: ExplainResponse | null
  isLoading: boolean
  error: string | null
  
  // Actions
  setSelectedTarget: (target: Target | null) => void
  setSmiles: (smiles: string) => void
  setSeed: (seed: number) => void
  setCalibrate: (calibrate: boolean) => void
  setAbstainBand: (band: [number, number]) => void
  setLastPrediction: (prediction: PredictResponse | null) => void
  setLastExplanation: (explanation: ExplainResponse | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  clearResults: () => void
}

interface AppState {
  // Global app state
  targets: Target[]
  metrics: MetricsResponse | null
  mockMode: boolean
  darkMode: boolean
  
  // Settings
  preferredModel: 'auto' | 'fusion' | 'baseline'
  defaultCalibrate: boolean
  defaultAbstainBand: [number, number]
  maxThreads: number
  batchSize: number
  colorBlindMode: boolean
  fontSize: 'sm' | 'md' | 'lg'
  
  // Actions
  setTargets: (targets: Target[]) => void
  setMetrics: (metrics: MetricsResponse | null) => void
  setMockMode: (mockMode: boolean) => void
  setDarkMode: (darkMode: boolean) => void
  setPreferredModel: (model: 'auto' | 'fusion' | 'baseline') => void
  setDefaultCalibrate: (calibrate: boolean) => void
  setDefaultAbstainBand: (band: [number, number]) => void
  setMaxThreads: (threads: number) => void
  setBatchSize: (size: number) => void
  setColorBlindMode: (enabled: boolean) => void
  setFontSize: (size: 'sm' | 'md' | 'lg') => void
}

interface BatchState {
  // Batch prediction state
  uploadedData: Array<{ target_id: string; smiles: string }> | null
  batchResults: Array<any> | null
  isProcessing: boolean
  progress: number
  rowsPerSecond: number
  eta: number
  errors: Array<{ idx: number; smiles: string; reason: string }>
  
  // Actions
  setUploadedData: (data: Array<{ target_id: string; smiles: string }> | null) => void
  setBatchResults: (results: Array<any> | null) => void
  setIsProcessing: (processing: boolean) => void
  setProgress: (progress: number) => void
  setRowsPerSecond: (rps: number) => void
  setEta: (eta: number) => void
  addError: (error: { idx: number; smiles: string; reason: string }) => void
  clearErrors: () => void
  clearBatch: () => void
}

// Prediction store (session state)
export const usePredictionStore = create<PredictionState>((set) => ({
  selectedTarget: null,
  smiles: '',
  seed: 42,
  calibrate: true,
  abstainBand: [0.45, 0.55],
  
  lastPrediction: null,
  lastExplanation: null,
  isLoading: false,
  error: null,
  
  setSelectedTarget: (target) => set({ selectedTarget: target }),
  setSmiles: (smiles) => set({ smiles }),
  setSeed: (seed) => set({ seed }),
  setCalibrate: (calibrate) => set({ calibrate }),
  setAbstainBand: (band) => set({ abstainBand: band }),
  setLastPrediction: (prediction) => set({ lastPrediction: prediction }),
  setLastExplanation: (explanation) => set({ lastExplanation: explanation }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  clearResults: () => set({ 
    lastPrediction: null, 
    lastExplanation: null, 
    error: null 
  }),
}))

// App store (persisted)
export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      targets: [],
      metrics: null,
      mockMode: false, // Use real API by default
      darkMode: false,
      
      preferredModel: 'auto',
      defaultCalibrate: true,
      defaultAbstainBand: [0.45, 0.55],
      maxThreads: 4,
      batchSize: 1000,
      colorBlindMode: false,
      fontSize: 'md',
      
      setTargets: (targets) => set({ targets }),
      setMetrics: (metrics) => set({ metrics }),
      setMockMode: (mockMode) => set({ mockMode }),
      setDarkMode: (darkMode) => set({ darkMode }),
      setPreferredModel: (model) => set({ preferredModel: model }),
      setDefaultCalibrate: (calibrate) => set({ defaultCalibrate: calibrate }),
      setDefaultAbstainBand: (band) => set({ defaultAbstainBand: band }),
      setMaxThreads: (threads) => set({ maxThreads: threads }),
      setBatchSize: (size) => set({ batchSize: size }),
      setColorBlindMode: (enabled) => set({ colorBlindMode: enabled }),
      setFontSize: (size) => set({ fontSize: size }),
    }),
    {
      name: 'mini-binding-settings',
      partialize: (state) => ({
        darkMode: state.darkMode,
        preferredModel: state.preferredModel,
        defaultCalibrate: state.defaultCalibrate,
        defaultAbstainBand: state.defaultAbstainBand,
        maxThreads: state.maxThreads,
        batchSize: state.batchSize,
        colorBlindMode: state.colorBlindMode,
        fontSize: state.fontSize,
      }),
    }
  )
)

// Batch store (session state)
export const useBatchStore = create<BatchState>((set) => ({
  uploadedData: null,
  batchResults: null,
  isProcessing: false,
  progress: 0,
  rowsPerSecond: 0,
  eta: 0,
  errors: [],
  
  setUploadedData: (data) => set({ uploadedData: data }),
  setBatchResults: (results) => set({ batchResults: results }),
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  setProgress: (progress) => set({ progress }),
  setRowsPerSecond: (rps) => set({ rowsPerSecond: rps }),
  setEta: (eta) => set({ eta }),
  addError: (error) => set((state) => ({ 
    errors: [...state.errors, error] 
  })),
  clearErrors: () => set({ errors: [] }),
  clearBatch: () => set({
    uploadedData: null,
    batchResults: null,
    isProcessing: false,
    progress: 0,
    rowsPerSecond: 0,
    eta: 0,
    errors: [],
  }),
}))

// Utility hooks
export const useCurrentPrediction = () => {
  const prediction = usePredictionStore((state) => state.lastPrediction)
  const explanation = usePredictionStore((state) => state.lastExplanation)
  const isLoading = usePredictionStore((state) => state.isLoading)
  const error = usePredictionStore((state) => state.error)
  
  return { prediction, explanation, isLoading, error }
}

export const useSettings = () => {
  const {
    preferredModel,
    defaultCalibrate,
    defaultAbstainBand,
    maxThreads,
    batchSize,
    colorBlindMode,
    fontSize,
    darkMode,
  } = useAppStore()
  
  return {
    preferredModel,
    defaultCalibrate,
    defaultAbstainBand,
    maxThreads,
    batchSize,
    colorBlindMode,
    fontSize,
    darkMode,
  }
}
