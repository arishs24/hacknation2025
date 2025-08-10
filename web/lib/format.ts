import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Number formatting utilities
export function formatProbability(prob: number): string {
  return (prob * 100).toFixed(1) + '%'
}

export function formatNumber(num: number, decimals: number = 3): string {
  return num.toFixed(decimals)
}

export function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`
  }
  return `${(ms / 1000).toFixed(1)}s`
}

export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`
}

// SMILES validation and processing
export function validateSmiles(smiles: string): { 
  valid: boolean
  message?: string
  atomCount?: number
  molecularWeight?: number
  logP?: number
} {
  // Basic SMILES validation - in production would use RDKit
  const trimmed = smiles.trim()
  
  if (!trimmed) {
    return { valid: false, message: 'SMILES cannot be empty' }
  }
  
  // Check for basic SMILES characters
  const validChars = /^[A-Za-z0-9()[\]@+\-=#$:./%\\]+$/
  if (!validChars.test(trimmed)) {
    return { valid: false, message: 'Contains invalid characters' }
  }
  
  // Basic bracket matching
  const brackets = { '(': ')', '[': ']' }
  const stack: string[] = []
  
  for (const char of trimmed) {
    if (char in brackets) {
      stack.push(brackets[char as keyof typeof brackets])
    } else if (Object.values(brackets).includes(char)) {
      if (stack.pop() !== char) {
        return { valid: false, message: 'Mismatched brackets' }
      }
    }
  }
  
  if (stack.length > 0) {
    return { valid: false, message: 'Unclosed brackets' }
  }
  
  // Mock atom count and properties (would compute with RDKit in production)
  const atomCount = (trimmed.match(/[A-Z]/g) || []).length
  const molecularWeight = atomCount * 12 + Math.random() * 200 // Mock
  const logP = Math.random() * 6 - 1 // Mock
  
  return { 
    valid: true, 
    atomCount,
    molecularWeight: Math.round(molecularWeight * 10) / 10,
    logP: Math.round(logP * 10) / 10
  }
}

export function sanitizeSmiles(smiles: string): string {
  // Basic sanitization - in production would use RDKit
  return smiles.trim().replace(/\s+/g, '')
}

export function standardizeSmiles(smiles: string): string {
  // Mock standardization - would use RDKit in production
  return sanitizeSmiles(smiles).toLowerCase()
}

// CSV processing utilities
export function parseCSV(csvText: string, hasHeaders: boolean = true): Array<Record<string, string>> {
  const lines = csvText.trim().split('\n')
  if (lines.length === 0) return []
  
  const headers = hasHeaders ? lines[0].split(',').map(h => h.trim()) : null
  const dataLines = hasHeaders ? lines.slice(1) : lines
  
  return dataLines.map((line, index) => {
    const values = line.split(',').map(v => v.trim())
    const row: Record<string, string> = {}
    
    if (headers) {
      headers.forEach((header, i) => {
        row[header] = values[i] || ''
      })
    } else {
      values.forEach((value, i) => {
        row[`col${i}`] = value
      })
    }
    
    row._index = index.toString()
    return row
  })
}

export function csvToArray(csvText: string): { target_id: string; smiles: string }[] {
  const parsed = parseCSV(csvText)
  return parsed.map(row => ({
    target_id: row.target_id || row.target || '',
    smiles: row.smiles || row.SMILES || ''
  }))
}

export function arrayToCSV(data: Array<Record<string, any>>): string {
  if (data.length === 0) return ''
  
  const headers = Object.keys(data[0])
  const csvLines = [headers.join(',')]
  
  data.forEach(row => {
    const values = headers.map(header => {
      const value = row[header]
      // Escape commas and quotes
      if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`
      }
      return value
    })
    csvLines.push(values.join(','))
  })
  
  return csvLines.join('\n')
}

// URL and query string utilities
export function updateQueryParams(params: Record<string, string | null>) {
  if (typeof window === 'undefined') return
  
  const url = new URL(window.location.href)
  
  Object.entries(params).forEach(([key, value]) => {
    if (value === null) {
      url.searchParams.delete(key)
    } else {
      url.searchParams.set(key, value)
    }
  })
  
  window.history.replaceState({}, '', url.toString())
}

export function getQueryParam(key: string): string | null {
  if (typeof window === 'undefined') return null
  
  const url = new URL(window.location.href)
  return url.searchParams.get(key)
}

// Confidence and threshold utilities
export function calculateConfidenceColor(probability: number, abstainBand: [number, number]): string {
  const [low, high] = abstainBand
  
  if (probability >= low && probability <= high) {
    return 'text-gray-500' // Abstain region
  }
  
  if (probability < 0.3 || probability > 0.7) {
    return 'text-green-600' // High confidence
  }
  
  return 'text-amber-600' // Medium confidence
}

export function calculateConfusionMatrix(
  predictions: number[],
  labels: number[],
  threshold: number
): { tp: number; fp: number; fn: number; tn: number } {
  let tp = 0, fp = 0, fn = 0, tn = 0
  
  predictions.forEach((pred, i) => {
    const predicted = pred >= threshold ? 1 : 0
    const actual = labels[i]
    
    if (predicted === 1 && actual === 1) tp++
    else if (predicted === 1 && actual === 0) fp++
    else if (predicted === 0 && actual === 1) fn++
    else tn++
  })
  
  return { tp, fp, fn, tn }
}

// Code generation utilities
export function generateCurlCommand(
  endpoint: string,
  method: string,
  body?: any,
  apiBase: string = 'http://localhost:8000'
): string {
  let command = `curl -X ${method} "${apiBase}${endpoint}"`
  
  if (body) {
    command += ` \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(body, null, 2)}'`
  }
  
  return command
}

export function generatePythonCode(
  endpoint: string,
  method: string,
  body?: any,
  apiBase: string = 'http://localhost:8000'
): string {
  let code = `import requests\nimport json\n\n`
  code += `url = "${apiBase}${endpoint}"\n`
  
  if (body) {
    code += `data = ${JSON.stringify(body, null, 2)}\n\n`
    code += `response = requests.${method.toLowerCase()}(url, json=data)\n`
  } else {
    code += `response = requests.${method.toLowerCase()}(url)\n`
  }
  
  code += `result = response.json()\nprint(json.dumps(result, indent=2))`
  
  return code
}

// Performance monitoring
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  waitMs: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => func.apply(null, args), waitMs)
  }
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  waitMs: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  
  return (...args: Parameters<T>) => {
    const now = Date.now()
    if (now - lastCall >= waitMs) {
      lastCall = now
      func.apply(null, args)
    }
  }
}

// Session export utilities
export function exportSession(data: {
  inputs: any
  outputs: any
  model: string
  temperature?: number
  timestamp: number
}): void {
  const sessionData = {
    ...data,
    version: '1.0',
    app: 'mini-binding-ui'
  }
  
  const blob = new Blob([JSON.stringify(sessionData, null, 2)], {
    type: 'application/json'
  })
  
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `binding-session-${new Date().toISOString().slice(0, 19)}.json`
  link.click()
  
  URL.revokeObjectURL(url)
}
