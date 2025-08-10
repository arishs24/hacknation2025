'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { useToast } from "@/components/ui/use-toast"
import { Upload, FileText, Play, Download, AlertCircle, CheckCircle } from "lucide-react"
import { cn } from "@/lib/format"
import { api } from "@/lib/api"
import { useAppStore } from "@/lib/store"
import type { PredictResponse } from "@/lib/schemas"

interface BatchResult {
  smiles: string
  target_id?: string
  prediction?: PredictResponse
  error?: string
}

export default function BatchPage() {
  const { toast } = useToast()
  const { targets } = useAppStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // File upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [fileData, setFileData] = useState<any[]>([])
  
  // Processing options
  const [selectedTarget, setSelectedTarget] = useState<string>("P24941") // CDK2 default
  const [standardizeSalts, setStandardizeSalts] = useState(true)
  const [skipInvalidRows, setSkipInvalidRows] = useState(true)
  const [calibrateOutputs, setCalibrateOutputs] = useState(true)
  const [enableAbstainBand, setEnableAbstainBand] = useState(false)
  const [enableOodCheck, setEnableOodCheck] = useState(true)
  const [cpuThreads, setCpuThreads] = useState([4])
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<BatchResult[]>([])
  const [progress, setProgress] = useState(0)

  const handleFileUpload = (file: File) => {
    if (!file) return
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      toast({
        title: "File too large",
        description: "Maximum file size is 10MB",
        variant: "destructive"
      })
      return
    }

    if (!file.name.endsWith('.csv')) {
      toast({
        title: "Invalid file type",
        description: "Please upload a CSV file",
        variant: "destructive"
      })
      return
    }

    setUploadedFile(file)
    
    // Parse CSV
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string
        const lines = text.split('\n').filter(line => line.trim())
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
        
        if (!headers.includes('smiles')) {
          toast({
            title: "Invalid CSV format",
            description: "CSV must contain a 'smiles' column",
            variant: "destructive"
          })
          return
        }
        
        const data = lines.slice(1).map(line => {
          const values = line.split(',').map(v => v.trim())
          const row: any = {}
          headers.forEach((header, idx) => {
            row[header] = values[idx] || ''
          })
          return row
        }).filter(row => row.smiles) // Filter out empty rows
        
        setFileData(data)
        toast({
          title: "File uploaded successfully",
          description: `Loaded ${data.length} compounds`,
          variant: "default"
        })
      } catch (error) {
        toast({
          title: "Parse error",
          description: "Failed to parse CSV file",
          variant: "destructive"
        })
      }
    }
    reader.readAsText(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const processBatch = async () => {
    if (!fileData.length) {
      toast({
        title: "No data to process",
        description: "Please upload a CSV file first",
        variant: "destructive"
      })
      return
    }

    setIsProcessing(true)
    setProgress(0)
    setResults([])

    try {
      // Prepare compounds for batch processing
      const compounds = fileData.map(row => ({
        smiles: row.smiles,
        target_id: row.target_id
      }))

      setProgress(50) // Show progress during API call

      const batchResponse = await api.batch(compounds, {
        target_id: selectedTarget,
        calibrate: calibrateOutputs,
        standardize_salts: standardizeSalts,
        skip_invalid: skipInvalidRows
      })

      if (batchResponse.success) {
        const batchResults: BatchResult[] = batchResponse.results.map((result: any) => ({
          smiles: result.smiles,
          target_id: result.target_id,
          prediction: result.success ? result.prediction : undefined,
          error: result.success ? undefined : result.error
        }))
        
        setResults(batchResults)
        setProgress(100)
        
        toast({
          title: "Batch processing complete",
          description: `Processed ${batchResults.length} compounds`,
          variant: "default"
        })
      } else {
        throw new Error(batchResponse.error || 'Batch processing failed')
      }
    } catch (error) {
      toast({
        title: "Batch processing failed",
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: "destructive"
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadResults = () => {
    if (!results.length) return
    
    const headers = ['smiles', 'target_id', 'binding_probability', 'pkd', 'model', 'calibrated', 'abstained', 'ood', 'error']
    const csvContent = [
      headers.join(','),
      ...results.map(result => [
        result.smiles,
        result.target_id || '',
        result.prediction?.proba?.toFixed(4) || '',
        result.prediction?.pkd?.toFixed(2) || '',
        result.prediction?.model || '',
        result.prediction?.calibrated || '',
        result.prediction?.abstained || '',
        result.prediction?.ood || '',
        result.error || ''
      ].join(','))
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `batch-predictions-${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Batch Processing</h1>
        <p className="text-muted-foreground mt-2">
          Upload CSV files for high-throughput binding predictions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload & Configure */}
        <Card>
          <CardHeader>
            <CardTitle>Upload & Configure</CardTitle>
            <CardDescription>
              Upload your CSV file and configure processing options
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Upload */}
            <div>
              <Label className="text-sm font-medium">CSV File Upload</Label>
              <div
                className={cn(
                  "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
                  "hover:border-primary/50 hover:bg-muted/50",
                  uploadedFile ? "border-green-500 bg-green-50" : "border-muted-foreground/25"
                )}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                {uploadedFile ? (
                  <div>
                    <p className="text-sm font-medium text-green-700">{uploadedFile.name}</p>
                    <p className="text-xs text-green-600 mt-1">{fileData.length} compounds loaded</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm">Click to upload or drag and drop</p>
                    <p className="text-xs text-muted-foreground mt-1">CSV with 'smiles' column (max 10MB)</p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              />
              <div className="text-xs text-muted-foreground mt-2 space-y-1">
                <p>• CSV format: columns should include 'smiles' and optionally 'target_id'</p>
                <p>• If no target_id provided, will default to CDK2 (P24941)</p>
                <p>• Maximum file size: 10MB</p>
              </div>
            </div>

            {/* Default Target Selection */}
            <div>
              <Label className="text-sm font-medium">Default Target (when target_id not provided)</Label>
              <Select value={selectedTarget} onValueChange={setSelectedTarget}>
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {targets.map((target) => (
                    <SelectItem key={target.target_id} value={target.target_id}>
                      {target.target_entry} ({target.target_id})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Processing Options */}
            <div>
              <Label className="text-sm font-medium mb-4 block">Processing Options</Label>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="standardize"
                    checked={standardizeSalts}
                    onCheckedChange={(checked) => setStandardizeSalts(checked === true)}
                  />
                  <Label htmlFor="standardize" className="text-sm">Standardize salts</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="skip-invalid"
                    checked={skipInvalidRows}
                    onCheckedChange={(checked) => setSkipInvalidRows(checked === true)}
                  />
                  <Label htmlFor="skip-invalid" className="text-sm">Skip invalid rows</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="calibrate"
                    checked={calibrateOutputs}
                    onCheckedChange={(checked) => setCalibrateOutputs(checked === true)}
                  />
                  <Label htmlFor="calibrate" className="text-sm">Calibrate outputs</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="abstain"
                    checked={enableAbstainBand}
                    onCheckedChange={(checked) => setEnableAbstainBand(checked === true)}
                  />
                  <Label htmlFor="abstain" className="text-sm">Enable abstain band</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="ood-check"
                    checked={enableOodCheck}
                    onCheckedChange={(checked) => setEnableOodCheck(checked === true)}
                  />
                  <Label htmlFor="ood-check" className="text-sm">Enable out-of-domain detection</Label>
                </div>
              </div>
            </div>

            {/* CPU Threads */}
            <div>
              <Label className="text-sm font-medium">CPU Threads</Label>
              <div className="mt-3">
                <Slider
                  value={cpuThreads}
                  onValueChange={setCpuThreads}
                  max={8}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>1</span>
                  <span>{cpuThreads[0]} (Current)</span>
                  <span>8</span>
                </div>
              </div>
            </div>

            {/* Run Button */}
            <Button
              onClick={processBatch}
              disabled={!fileData.length || isProcessing}
              className="w-full"
              size="lg"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Processing... ({progress.toFixed(0)}%)
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Batch
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Processing Results */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Processing Results</span>
              <div className="flex gap-2">
                {results.length > 0 && (
                  <Button variant="outline" size="sm" onClick={downloadResults}>
                    <Download className="h-4 w-4 mr-1" />
                    Download CSV
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isProcessing ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
                <p className="text-sm text-muted-foreground">Processing batch predictions...</p>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-4">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-2">{progress.toFixed(0)}% complete</p>
              </div>
            ) : results.length > 0 ? (
              <div className="space-y-4">
                <div className="text-sm text-muted-foreground">
                  Processed {results.length} compounds
                </div>
                <div className="max-h-96 overflow-y-auto">
                  <div className="space-y-2">
                    {results.slice(0, 50).map((result, idx) => (
                      <div
                        key={idx}
                        className={cn(
                          "p-3 rounded border text-sm",
                          result.error ? "border-red-200 bg-red-50" : "border-green-200 bg-green-50"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {result.error ? (
                              <AlertCircle className="h-4 w-4 text-red-500" />
                            ) : (
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            )}
                            <span className="font-mono text-xs">{result.smiles.slice(0, 30)}...</span>
                          </div>
                          {result.prediction && (
                            <div className="text-right">
                              <div className="font-medium">
                                {(result.prediction.proba * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs text-muted-foreground">
                                pKd: {result.prediction.pkd?.toFixed(2) || 'N/A'}
                              </div>
                            </div>
                          )}
                        </div>
                        {result.error && (
                          <div className="text-xs text-red-600 mt-1">{result.error}</div>
                        )}
                      </div>
                    ))}
                    {results.length > 50 && (
                      <div className="text-center text-xs text-muted-foreground py-2">
                        ... and {results.length - 50} more results
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <FileText className="h-8 w-8 mx-auto mb-2" />
                <p>No results to display</p>
                <p className="text-xs mt-2">Upload a CSV file and run batch processing</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
