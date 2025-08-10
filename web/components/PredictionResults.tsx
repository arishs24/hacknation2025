import * as React from "react"
import { AlertTriangle, Clock, Cpu, Download, Copy, Thermometer, CheckCircle } from "lucide-react"
import { type PredictResponse, type ExplainResponse, type Target } from "@/lib/schemas"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { formatProbability, formatLatency, formatNumber } from "@/lib/format"
import { cn } from "@/lib/format"
import { useToast } from "@/components/ui/use-toast"
import html2canvas from "html2canvas"

interface PredictionResultsProps {
  prediction: PredictResponse | null
  explanation: ExplainResponse | null
  isLoading: boolean
  error: string | null
  target: Target | null
  smiles: string
}

export function PredictionResults({
  prediction,
  explanation,
  isLoading,
  error,
  target,
  smiles
}: PredictionResultsProps) {
  const { toast } = useToast()
  const resultsRef = React.useRef<HTMLDivElement>(null)
  const [isExporting, setIsExporting] = React.useState(false)
  const [isCopying, setIsCopying] = React.useState(false)

  // Export results as PNG
  const exportToPNG = async () => {
    if (!resultsRef.current || !prediction) {
      toast({
        title: "Export failed",
        description: "No results to export",
        variant: "destructive"
      })
      return
    }

    setIsExporting(true)
    try {
      const canvas = await html2canvas(resultsRef.current, {
        backgroundColor: '#ffffff',
        scale: 2, // Higher quality
        useCORS: true,
        allowTaint: true,
        logging: false
      })

      // Create download link
      const link = document.createElement('a')
      link.download = `binding-prediction-${target?.target_entry || 'result'}-${Date.now()}.png`
      link.href = canvas.toDataURL('image/png')
      link.click()

      toast({
        title: "Export successful",
        description: "PNG file has been downloaded",
        variant: "default"
      })
    } catch (error) {
      console.error('Export failed:', error)
      toast({
        title: "Export failed",
        description: "Failed to generate PNG file",
        variant: "destructive"
      })
    } finally {
      setIsExporting(false)
    }
  }

  // Copy API request as cURL
  const copyAsCurl = async () => {
    if (!target || !smiles || !prediction) {
      toast({
        title: "Copy failed",
        description: "No prediction data available",
        variant: "destructive"
      })
      return
    }

    setIsCopying(true)
    const curlCommand = `# Binding Prediction API Example
# Prediction endpoint
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "target_id": "${target.target_id}",
    "smiles": "${smiles}",
    "calibrate": true,
    "seed": 42
  }'

# Scientific Analysis endpoint
curl -X POST "http://localhost:8000/explain" \\
  -H "Content-Type: application/json" \\
  -d '{
    "target_id": "${target.target_id}",
    "smiles": "${smiles}"
  }'`

    try {
      await navigator.clipboard.writeText(curlCommand)
      toast({
        title: "Copied to clipboard",
        description: "cURL command has been copied",
        variant: "default"
      })
    } catch (error) {
      console.error('Copy failed:', error)
      // Fallback for older browsers
      const textArea = document.createElement('textarea')
      textArea.value = curlCommand
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
      
      toast({
        title: "Copied to clipboard",
        description: "cURL command has been copied",
        variant: "default"
      })
    } finally {
      setIsCopying(false)
    }
  }
  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center space-y-2">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto" />
            <p className="text-muted-foreground">Running prediction...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center space-y-2">
            <AlertTriangle className="h-8 w-8 text-destructive mx-auto" />
            <p className="text-destructive font-medium">Prediction Failed</p>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!prediction) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center space-y-2">
            <div className="h-8 w-8 rounded-full bg-muted mx-auto flex items-center justify-center">
              <span className="text-muted-foreground">?</span>
            </div>
            <p className="text-muted-foreground">No prediction yet</p>
            <p className="text-xs text-muted-foreground">
              Select a target and enter SMILES to get started
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4" ref={resultsRef}>
      {/* Main probability result */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Binding Probability</span>
            <div className="flex items-center gap-2">
              {prediction.calibrated && (
                <div className="flex items-center gap-1 text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                  <Thermometer className="h-3 w-3" />
                  T = {formatNumber(prediction.temperature || 1, 2)}
                </div>
              )}
              <div className={cn(
                "text-xs px-2 py-1 rounded",
                prediction.model === 'fusion' 
                  ? "bg-blue-100 text-blue-800" 
                  : "bg-amber-100 text-amber-800"
              )}>
                {prediction.model}
              </div>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Probability display */}
            <div className="text-center">
              {prediction.abstained ? (
                <div className="space-y-2">
                  <div className="text-4xl font-bold text-muted-foreground">—</div>
                  <p className="text-sm text-muted-foreground">
                    Model abstains: low confidence
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-4xl font-bold">
                    {formatProbability(prediction.proba)}
                  </div>
                  {prediction.calibrated && (
                    <p className="text-xs text-green-600">
                      Calibrated (ECE ≤ 0.08)
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Confidence bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
              <div className="relative h-3 bg-gradient-to-r from-red-200 via-yellow-200 to-green-200 rounded">
                <div
                  className="absolute top-0 h-full w-1 bg-foreground rounded"
                  style={{ left: `${prediction.proba * 100}%` }}
                />
              </div>
            </div>

            {/* Additional info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              {prediction.pkd && (
                <div>
                  <span className="text-muted-foreground">pKd:</span>
                  <span className="ml-1 font-medium">{formatNumber(prediction.pkd, 1)}</span>
                </div>
              )}
              <div>
                <span className="text-muted-foreground">Latency:</span>
                <span className="ml-1 font-medium">{formatLatency(prediction.latency_ms)}</span>
              </div>
            </div>

            {/* Warning chips */}
            <div className="flex gap-2">
              <div className={cn(
                "flex items-center gap-1 text-xs px-2 py-1 rounded",
                prediction.abstained 
                  ? "bg-gray-100 text-gray-800 border border-gray-200" 
                  : "bg-green-100 text-green-600"
              )}>
                <Clock className="h-3 w-3" />
                {prediction.abstained ? "Abstained" : "Confident"}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action buttons */}
      <div className="grid grid-cols-2 gap-2">
        <Button variant="outline" size="sm" onClick={exportToPNG} disabled={isExporting}>
          {isExporting ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-1" />
          ) : (
            <Download className="h-4 w-4 mr-1" />
          )}
          {isExporting ? "Exporting..." : "Export PNG"}
        </Button>
        <Button variant="outline" size="sm" onClick={copyAsCurl} disabled={isCopying}>
          {isCopying ? (
            <CheckCircle className="h-4 w-4 mr-1 text-green-600" />
          ) : (
            <Copy className="h-4 w-4 mr-1" />
          )}
          {isCopying ? "Copied!" : "Copy as cURL"}
        </Button>
      </div>

      {/* Scientific Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Scientific Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          {explanation ? (
            <div className="space-y-6">
              {/* Molecular Properties */}
              <div>
                <h3 className="font-semibold mb-3">Molecular Properties</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Molecular Weight:</span>
                    <div className="font-medium">{explanation.molecular_properties.molecular_weight} Da</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">LogP:</span>
                    <div className="font-medium">{explanation.molecular_properties.logp}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">H-bond Donors:</span>
                    <div className="font-medium">{explanation.molecular_properties.hbd}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">H-bond Acceptors:</span>
                    <div className="font-medium">{explanation.molecular_properties.hba}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">TPSA:</span>
                    <div className="font-medium">{explanation.molecular_properties.tpsa} Ų</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Rotatable Bonds:</span>
                    <div className="font-medium">{explanation.molecular_properties.rotatable_bonds}</div>
                  </div>
                </div>
                <div className="mt-3 flex items-center space-x-4">
                  <div>
                    <span className="text-muted-foreground">Drug-likeness Score:</span>
                    <div className={cn("font-medium", 
                      explanation.molecular_properties.drug_likeness_score >= 0.7 ? "text-green-600" : 
                      explanation.molecular_properties.drug_likeness_score >= 0.4 ? "text-yellow-600" : 
                      "text-red-600"
                    )}>
                      {explanation.molecular_properties.drug_likeness_score.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Lipinski Violations:</span>
                    <div className={cn("font-medium",
                      explanation.molecular_properties.lipinski_violations === 0 ? "text-green-600" :
                      explanation.molecular_properties.lipinski_violations <= 1 ? "text-yellow-600" :
                      "text-red-600"
                    )}>
                      {explanation.molecular_properties.lipinski_violations}
                    </div>
                  </div>
                </div>
              </div>

              {/* Binding Analysis */}
              <div>
                <h3 className="font-semibold mb-3">Binding Analysis</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-4">
                    <div>
                      <span className="text-muted-foreground">Affinity Class:</span>
                      <div className={cn("font-medium",
                        explanation.binding_analysis.binding_affinity_class === "Strong" ? "text-green-600" :
                        explanation.binding_analysis.binding_affinity_class === "Moderate" ? "text-yellow-600" :
                        "text-red-600"
                      )}>
                        {explanation.binding_analysis.binding_affinity_class}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Confidence:</span>
                      <div className={cn("font-medium",
                        explanation.binding_analysis.confidence_level === "High" ? "text-green-600" :
                        explanation.binding_analysis.confidence_level === "Medium" ? "text-yellow-600" :
                        "text-red-600"
                      )}>
                        {explanation.binding_analysis.confidence_level}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Binding Mode:</span>
                      <div className="font-medium">{explanation.binding_analysis.binding_mode}</div>
                    </div>
                  </div>
                  
                  {explanation.binding_analysis.key_interactions.length > 0 && (
                    <div>
                      <span className="text-muted-foreground block mb-2">Key Interactions:</span>
                      <ul className="text-sm space-y-1">
                        {explanation.binding_analysis.key_interactions.map((interaction, idx) => (
                          <li key={idx} className="flex items-center space-x-2">
                            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full" />
                            <span>{interaction}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>

              {/* Structural Alerts */}
              {explanation.structural_alerts.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-3 text-red-600">Structural Alerts</h3>
                  <ul className="text-sm space-y-1">
                    {explanation.structural_alerts.map((alert, idx) => (
                      <li key={idx} className="flex items-center space-x-2 text-red-600">
                        <AlertTriangle className="h-4 w-4" />
                        <span>{alert}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Optimization Suggestions */}
              {explanation.optimization_suggestions.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-3">Optimization Suggestions</h3>
                  <ul className="text-sm space-y-1">
                    {explanation.optimization_suggestions.map((suggestion, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                        <span>{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Chemical Novelty Analysis */}
              {explanation.chemical_novelty_analysis && explanation.chemical_novelty_analysis.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-3">Chemical Novelty Analysis</h3>
                  <ul className="text-sm space-y-1">
                    {explanation.chemical_novelty_analysis.map((insight, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Confidence Score */}
              <div className="pt-3 border-t">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Overall Confidence:</span>
                  <div className={cn("font-medium text-lg",
                    explanation.confidence_score >= 0.7 ? "text-green-600" :
                    explanation.confidence_score >= 0.4 ? "text-yellow-600" :
                    "text-red-600"
                  )}>
                    {explanation.confidence_score.toFixed(3)}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              <p>Run a prediction to see molecular analysis</p>
              <p className="text-xs mt-2">
                Includes drug-likeness, binding analysis, chemical novelty insights, and optimization suggestions
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
