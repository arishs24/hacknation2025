'use client'

import { useState, useEffect } from 'react'
import { usePredictionStore, useAppStore } from '@/lib/store'
import { updateQueryParams } from '@/lib/format'
import { TargetSelector } from '@/components/TargetSelector'
import { SmilesInput } from '@/components/SmilesInput'
import { InferenceControls } from '@/components/InferenceControls'
import { PredictionResults } from '@/components/PredictionResults'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { useToast } from '@/components/ui/use-toast'

export default function PredictPage() {
  const { toast } = useToast()
  
  const {
    selectedTarget,
    smiles,
    seed,
    calibrate,
    abstainBand,
    lastPrediction,
    lastExplanation,
    isLoading,
    error,
    setSelectedTarget,
    setSmiles,
    setSeed,
    setCalibrate,
    setAbstainBand,
    setLastPrediction,
    setLastExplanation,
    setLoading,
    setError,
    clearResults
  } = usePredictionStore()

  const { targets, setTargets } = useAppStore()

  // Load targets on mount
  useEffect(() => {
    async function loadTargets() {
      try {
        const targetData = await api.targets()
        setTargets(targetData)
      } catch (err) {
        console.error('Failed to load targets:', err)
        toast({
          title: "Failed to load targets",
          description: "Could not fetch available protein targets",
          variant: "destructive"
        })
      }
    }
    
    if (targets.length === 0) {
      loadTargets()
    }
  }, [targets.length, setTargets, toast])

  const handlePredict = async () => {
    if (!selectedTarget || !smiles.trim()) {
      toast({
        title: "Missing inputs",
        description: "Please select a target and enter a SMILES string",
        variant: "destructive"
      })
      return
    }

    setLoading(true)
    setError(null)
    clearResults()
    
    // Force clear explanation immediately to prevent stale data
    setLastExplanation(null)

    try {
      // Get prediction
      const prediction = await api.predict({
        target_id: selectedTarget.target_id,
        smiles: smiles.trim(),
        seed,
        calibrate,
        enable_ood_check: true
      })

      setLastPrediction(prediction)
      
      // Get scientific analysis/explanation
      try {
        console.log('Requesting explanation for:', {
          target_id: selectedTarget.target_id,
          smiles: smiles.trim(),
          timestamp: new Date().toISOString()
        })
        const explanation = await api.explain({
          target_id: selectedTarget.target_id,
          smiles: smiles.trim()
        })
        console.log('Fresh explanation received:', {
          molecular_weight: explanation.molecular_properties.molecular_weight,
          logp: explanation.molecular_properties.logp,
          timestamp: new Date().toISOString()
        })
        setLastExplanation(explanation)
      } catch (explainError) {
        console.error('Failed to get explanation:', explainError)
        // Don't fail the whole prediction if explanation fails
        setLastExplanation(null)
      }
      
      if (prediction.abstained) {
        toast({
          title: "Model abstained",
          description: "Prediction confidence is in the abstain band",
          variant: "default"
        })
      } else {
        toast({
          title: "Prediction complete",
          description: `Binding probability: ${(prediction.proba * 100).toFixed(1)}%`,
          variant: "default"
        })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed'
      setError(message)
      toast({
        title: "Prediction failed",
        description: message,
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }

  const canPredict = selectedTarget && smiles.trim() && !isLoading

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
      {/* Left Column - Inputs */}
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Target Selection</CardTitle>
            <CardDescription>
              Choose from top-10 human kinases
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TargetSelector
              targets={targets}
              value={selectedTarget}
              onChange={setSelectedTarget}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Molecule Input</CardTitle>
            <CardDescription>
              Enter SMILES string for binding prediction
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SmilesInput
              value={smiles}
              onChange={setSmiles}
              onSanitize={() => {
                // Basic sanitization in mock mode
                setSmiles(smiles.trim())
              }}
              onStandardize={() => {
                // Basic standardization in mock mode
                setSmiles(smiles.trim().toLowerCase())
              }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Inference Controls</CardTitle>
            <CardDescription>
              Configure prediction parameters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <InferenceControls
              calibrate={calibrate}
              onCalibrateChange={setCalibrate}
              abstainBand={abstainBand}
              onAbstainBandChange={setAbstainBand}
              seed={seed}
              onSeedChange={setSeed}
            />
          </CardContent>
        </Card>

        <Button
          onClick={handlePredict}
          disabled={!canPredict}
          className="w-full"
          size="lg"
        >
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Predict Binding (⌘⏎)
        </Button>
      </div>

      {/* Right Column - Results */}
      <div className="space-y-6">
        <PredictionResults
          prediction={lastPrediction}
          explanation={lastExplanation}
          isLoading={isLoading}
          error={error}
          target={selectedTarget}
          smiles={smiles}
        />
      </div>
    </div>
  )
}

// Keyboard shortcuts
if (typeof window !== 'undefined') {
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      const predictButton = document.querySelector('button[data-predict]') as HTMLButtonElement
      if (predictButton && !predictButton.disabled) {
        predictButton.click()
      }
    }
  })
}
