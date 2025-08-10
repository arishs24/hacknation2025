"use client"

import * as React from "react"
import { Switch } from "@radix-ui/react-switch"
import { Thermometer, Dice6, Settings } from "lucide-react"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/format"

interface InferenceControlsProps {
  calibrate: boolean
  onCalibrateChange: (value: boolean) => void
  abstainBand: [number, number]
  onAbstainBandChange: (value: [number, number]) => void
  seed: number
  onSeedChange: (value: number) => void
}

export function InferenceControls({
  calibrate,
  onCalibrateChange,
  abstainBand,
  onAbstainBandChange,
  seed,
  onSeedChange
}: InferenceControlsProps) {
  const [localAbstainLow, setLocalAbstainLow] = React.useState(abstainBand[0])
  const [localAbstainHigh, setLocalAbstainHigh] = React.useState(abstainBand[1])

  // Update parent when local values change
  React.useEffect(() => {
    const low = Math.min(localAbstainLow, localAbstainHigh)
    const high = Math.max(localAbstainLow, localAbstainHigh)
    onAbstainBandChange([low, high])
  }, [localAbstainLow, localAbstainHigh, onAbstainBandChange])

  const abstainPercentage = ((abstainBand[1] - abstainBand[0]) * 100).toFixed(1)

  return (
    <div className="space-y-6">
      {/* Calibration toggle */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <Label htmlFor="calibrate-toggle" className="flex items-center gap-2">
            <Thermometer className="h-4 w-4" />
            Temperature Calibration
          </Label>
          <p className="text-xs text-muted-foreground">
            Apply temperature scaling for calibrated probabilities
          </p>
        </div>
        <Switch
          id="calibrate-toggle"
          checked={calibrate}
          onCheckedChange={onCalibrateChange}
          className="data-[state=checked]:bg-primary"
        />
      </div>

      {calibrate && (
        <div className="pl-6 text-xs text-muted-foreground border-l-2 border-muted">
          When enabled, temperature scaling will be applied to improve probability calibration.
          The temperature parameter (T) will be displayed with results.
        </div>
      )}

      {/* Abstain band configuration */}
      <div className="space-y-3">
        <Label className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          Abstain Band
        </Label>
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label htmlFor="abstain-low" className="text-xs">Lower bound</Label>
            <Input
              id="abstain-low"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={localAbstainLow}
              onChange={(e) => setLocalAbstainLow(parseFloat(e.target.value) || 0)}
              className="text-xs"
            />
          </div>
          <div>
            <Label htmlFor="abstain-high" className="text-xs">Upper bound</Label>
            <Input
              id="abstain-high"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={localAbstainHigh}
              onChange={(e) => setLocalAbstainHigh(parseFloat(e.target.value) || 1)}
              className="text-xs"
            />
          </div>
        </div>

        {/* Visual representation of abstain band */}
        <div className="space-y-2">
          <div className="relative h-6 bg-gradient-to-r from-red-200 via-yellow-200 to-green-200 rounded">
            <div
              className="absolute top-0 h-full bg-gray-400 opacity-50 rounded"
              style={{
                left: `${abstainBand[0] * 100}%`,
                width: `${(abstainBand[1] - abstainBand[0]) * 100}%`
              }}
            />
            <div className="absolute inset-0 flex items-center justify-between px-1 text-xs font-medium">
              <span>0</span>
              <span>0.5</span>
              <span>1</span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Predictions in the gray band ({abstainBand[0].toFixed(2)} - {abstainBand[1].toFixed(2)}) 
            will be marked as abstained. Band width: {abstainPercentage}%
          </p>
        </div>
      </div>

      {/* Random seed */}
      <div className="space-y-2">
        <Label htmlFor="seed-input" className="flex items-center gap-2">
          <Dice6 className="h-4 w-4" />
          Random Seed
        </Label>
        <Input
          id="seed-input"
          type="number"
          min="0"
          max="999999"
          value={seed}
          onChange={(e) => onSeedChange(parseInt(e.target.value) || 42)}
          className="w-24"
        />
        <p className="text-xs text-muted-foreground">
          Seed for reproducible predictions. Default: 42
        </p>
      </div>
    </div>
  )
}
