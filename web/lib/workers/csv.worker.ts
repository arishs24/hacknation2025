// CSV processing worker for handling large files without blocking the UI

interface ProcessMessage {
  type: 'process'
  csvText: string
  chunkSize?: number
}

interface ProgressMessage {
  type: 'progress'
  processed: number
  total: number
}

interface ResultMessage {
  type: 'result'
  data: Array<{ target_id: string; smiles: string; _index: number }>
}

interface ErrorMessage {
  type: 'error'
  message: string
}

type WorkerMessage = ProcessMessage
type WorkerResponse = ProgressMessage | ResultMessage | ErrorMessage

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++ // Skip next quote
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }
  
  result.push(current.trim())
  return result
}

function validateSmilesBasic(smiles: string): boolean {
  // Basic validation without RDKit
  if (!smiles || smiles.trim().length === 0) return false
  
  // Check for basic SMILES characters
  const validChars = /^[A-Za-z0-9()[\]@+\-=#$:./%\\]+$/
  return validChars.test(smiles.trim())
}

function processCSVChunk(
  lines: string[],
  headers: string[] | null,
  startIndex: number
): Array<{ target_id: string; smiles: string; _index: number; valid: boolean }> {
  const results: Array<{ target_id: string; smiles: string; _index: number; valid: boolean }> = []
  
  lines.forEach((line, lineIndex) => {
    if (!line.trim()) return // Skip empty lines
    
    const values = parseCSVLine(line)
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
    
    const target_id = row.target_id || row.target || row.Target || ''
    const smiles = row.smiles || row.SMILES || row.Smiles || ''
    const valid = validateSmilesBasic(smiles) && target_id.length > 0
    
    results.push({
      target_id,
      smiles: smiles.trim(),
      _index: startIndex + lineIndex,
      valid
    })
  })
  
  return results
}

async function processCSV(csvText: string, chunkSize: number = 1000): Promise<void> {
  try {
    const lines = csvText.trim().split('\n')
    if (lines.length === 0) {
      self.postMessage({ type: 'error', message: 'Empty CSV file' } as ErrorMessage)
      return
    }
    
    // Detect headers
    const firstLine = lines[0]
    const hasHeaders = /smiles|target/i.test(firstLine)
    
    const headers = hasHeaders ? parseCSVLine(firstLine) : null
    const dataLines = hasHeaders ? lines.slice(1) : lines
    
    if (dataLines.length === 0) {
      self.postMessage({ type: 'error', message: 'No data rows found' } as ErrorMessage)
      return
    }
    
    const allResults: Array<{ target_id: string; smiles: string; _index: number }> = []
    const totalLines = dataLines.length
    
    // Process in chunks to avoid blocking
    for (let i = 0; i < dataLines.length; i += chunkSize) {
      const chunk = dataLines.slice(i, i + chunkSize)
      const chunkResults = processCSVChunk(chunk, headers, i)
      
      // Filter valid results and add to final array
      const validResults = chunkResults
        .filter(r => r.valid)
        .map(r => ({ target_id: r.target_id, smiles: r.smiles, _index: r._index }))
      
      allResults.push(...validResults)
      
      // Send progress update
      self.postMessage({
        type: 'progress',
        processed: i + chunk.length,
        total: totalLines
      } as ProgressMessage)
      
      // Yield control to prevent blocking
      await new Promise(resolve => setTimeout(resolve, 0))
    }
    
    // Send final results
    self.postMessage({
      type: 'result',
      data: allResults
    } as ResultMessage)
    
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : 'Unknown error processing CSV'
    } as ErrorMessage)
  }
}

// Worker message handler
self.onmessage = function(event: MessageEvent<WorkerMessage>) {
  const { type, csvText, chunkSize = 1000 } = event.data
  
  if (type === 'process') {
    processCSV(csvText, chunkSize)
  }
}

// Export type for TypeScript
export type { WorkerMessage, WorkerResponse }
