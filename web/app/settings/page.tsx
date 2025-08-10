'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { useToast } from "@/components/ui/use-toast"
import { Settings, Cpu, Database, RefreshCw, Trash2 } from "lucide-react"
import { cn } from "@/lib/format"
import { useAppStore } from "@/lib/store"

interface SettingsConfig {
  // Model Configuration
  defaultModel: string
  enableCalibration: boolean
  abstainThreshold: number
  
  // Featurization
  ligandFeatures: string
  proteinFeatures: string
  
  // Performance
  cpuThreads: number
  runFullyOffline: boolean
  
  // Application
  theme: string
  autoSave: boolean
  showAdvancedOptions: boolean
  decimals: number
}

const DEFAULT_SETTINGS: SettingsConfig = {
  defaultModel: 'fusion-mlp',
  enableCalibration: true,
  abstainThreshold: 0.1,
  ligandFeatures: 'ecfp4-2048',
  proteinFeatures: 'esm-tiny',
  cpuThreads: 4,
  runFullyOffline: false,
  theme: 'system',
  autoSave: true,
  showAdvancedOptions: false,
  decimals: 3
}

export default function SettingsPage() {
  const { toast } = useToast()
  const { targets } = useAppStore()
  
  const [settings, setSettings] = useState<SettingsConfig>(DEFAULT_SETTINGS)
  const [hasChanges, setHasChanges] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('mini-binding-settings')
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings)
        setSettings({ ...DEFAULT_SETTINGS, ...parsed })
      } catch (error) {
        console.error('Failed to parse saved settings:', error)
      }
    }
  }, [])

  const updateSetting = <K extends keyof SettingsConfig>(
    key: K,
    value: SettingsConfig[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const saveSettings = () => {
    setIsLoading(true)
    try {
      localStorage.setItem('mini-binding-settings', JSON.stringify(settings))
      setHasChanges(false)
      toast({
        title: "Settings saved",
        description: "Your preferences have been saved successfully",
        variant: "default"
      })
    } catch (error) {
      toast({
        title: "Save failed",
        description: "Failed to save settings",
        variant: "destructive"
      })
    } finally {
      setIsLoading(false)
    }
  }

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS)
    setHasChanges(true)
    toast({
      title: "Settings reset",
      description: "All settings have been reset to defaults",
      variant: "default"
    })
  }

  const clearCaches = () => {
    // Clear relevant localStorage items
    localStorage.removeItem('mini-binding-cache')
    localStorage.removeItem('mini-binding-predictions')
    toast({
      title: "Caches cleared",
      description: "All cached data has been removed",
      variant: "default"
    })
  }

  const clearAllData = () => {
    // Clear all application data
    localStorage.clear()
    setSettings(DEFAULT_SETTINGS)
    setHasChanges(false)
    toast({
      title: "Application reset",
      description: "All settings and cached data have been cleared",
      variant: "default"
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-2">
            Configure model behavior and application preferences
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={resetSettings}
            disabled={isLoading}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button
            onClick={saveSettings}
            disabled={!hasChanges || isLoading}
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Settings className="h-4 w-4 mr-2" />
                Save Settings
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Model Configuration</CardTitle>
            <CardDescription>
              Default model selection and prediction behavior
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Model Selection */}
            <div>
              <Label className="text-sm font-medium">Default Model</Label>
              <Select
                value={settings.defaultModel}
                onValueChange={(value) => updateSetting('defaultModel', value)}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fusion-mlp">Fusion MLP (Recommended)</SelectItem>
                  <SelectItem value="baseline-lr">Baseline Logistic Regression</SelectItem>
                  <SelectItem value="ensemble">Ensemble (Auto-select)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Calibration */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="calibration"
                checked={settings.enableCalibration}
                onCheckedChange={(checked) => updateSetting('enableCalibration', checked === true)}
              />
              <Label htmlFor="calibration" className="text-sm">
                Apply temperature scaling (T=1.2)
              </Label>
            </div>

            {/* Abstain Band */}
            <div>
              <Label className="text-sm font-medium">
                Abstain Threshold ({settings.abstainThreshold.toFixed(2)})
              </Label>
              <div className="mt-3">
                <Slider
                  value={[settings.abstainThreshold]}
                  onValueChange={([value]) => updateSetting('abstainThreshold', value)}
                  max={0.5}
                  min={0.0}
                  step={0.01}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>0.0 (Never abstain)</span>
                  <span>0.5 (Conservative)</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Featurization */}
        <Card>
          <CardHeader>
            <CardTitle>Featurization</CardTitle>
            <CardDescription>
              Molecular and protein representation methods
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Ligand Features */}
            <div>
              <Label className="text-sm font-medium">Ligand Features</Label>
              <Select
                value={settings.ligandFeatures}
                onValueChange={(value) => updateSetting('ligandFeatures', value)}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ecfp4-2048">ECFP4 (2048 bits)</SelectItem>
                  <SelectItem value="ecfp4-1024">ECFP4 (1024 bits)</SelectItem>
                  <SelectItem value="rdkit-descriptors">RDKit Descriptors</SelectItem>
                  <SelectItem value="morgan-2048">Morgan Fingerprints (2048 bits)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Protein Features */}
            <div>
              <Label className="text-sm font-medium">Protein Features</Label>
              <Select
                value={settings.proteinFeatures}
                onValueChange={(value) => updateSetting('proteinFeatures', value)}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="esm-tiny">ESM-tiny (frozen)</SelectItem>
                  <SelectItem value="esm-small">ESM-small (8M params)</SelectItem>
                  <SelectItem value="unirep">UniRep embeddings</SelectItem>
                  <SelectItem value="onehot">One-hot encoding</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Performance
            </CardTitle>
            <CardDescription>
              Computational and runtime optimization settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* CPU Threads */}
            <div>
              <Label className="text-sm font-medium">
                CPU Threads ({settings.cpuThreads})
              </Label>
              <div className="mt-3">
                <Slider
                  value={[settings.cpuThreads]}
                  onValueChange={([value]) => updateSetting('cpuThreads', value)}
                  max={16}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>1</span>
                  <span>Current: {settings.cpuThreads}</span>
                  <span>16</span>
                </div>
              </div>
            </div>

            {/* Offline Mode */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="offline"
                checked={settings.runFullyOffline}
                onCheckedChange={(checked) => updateSetting('runFullyOffline', checked === true)}
              />
              <Label htmlFor="offline" className="text-sm">
                Disable all network calls
              </Label>
            </div>
          </CardContent>
        </Card>

        {/* Application Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Application Settings</CardTitle>
            <CardDescription>
              User interface and behavior preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Theme */}
            <div>
              <Label className="text-sm font-medium">Theme</Label>
              <Select
                value={settings.theme}
                onValueChange={(value) => updateSetting('theme', value)}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="light">Light</SelectItem>
                  <SelectItem value="dark">Dark</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Decimal Places */}
            <div>
              <Label className="text-sm font-medium">
                Decimal Places ({settings.decimals})
              </Label>
              <div className="mt-3">
                <Slider
                  value={[settings.decimals]}
                  onValueChange={([value]) => updateSetting('decimals', value)}
                  max={6}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>1</span>
                  <span>6</span>
                </div>
              </div>
            </div>

            {/* Auto-save */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="autosave"
                checked={settings.autoSave}
                onCheckedChange={(checked) => updateSetting('autoSave', checked === true)}
              />
              <Label htmlFor="autosave" className="text-sm">
                Auto-save predictions
              </Label>
            </div>

            {/* Advanced Options */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="advanced"
                checked={settings.showAdvancedOptions}
                onCheckedChange={(checked) => updateSetting('showAdvancedOptions', checked === true)}
              />
              <Label htmlFor="advanced" className="text-sm">
                Show advanced options
              </Label>
            </div>
          </CardContent>
        </Card>

        {/* Data Management */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Management
            </CardTitle>
            <CardDescription>
              Manage cached data, predictions, and application state
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Clear Caches */}
              <div className="p-4 border rounded-lg">
                <h3 className="font-medium mb-2">Clear Caches</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Remove temporary files and cached predictions
                </p>
                <Button
                  variant="outline"
                  onClick={clearCaches}
                  className="w-full"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Caches
                </Button>
              </div>

              {/* Reset Application */}
              <div className="p-4 border rounded-lg border-orange-200">
                <h3 className="font-medium mb-2 text-orange-700">Reset Application</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Clear all settings and cached data
                </p>
                <Button
                  variant="outline"
                  onClick={clearAllData}
                  className="w-full border-orange-300 text-orange-700 hover:bg-orange-50"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reset App
                </Button>
              </div>

              {/* Export Settings */}
              <div className="p-4 border rounded-lg border-blue-200">
                <h3 className="font-medium mb-2 text-blue-700">Export Settings</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Download current configuration as JSON
                </p>
                <Button
                  variant="outline"
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(settings, null, 2)], {
                      type: 'application/json'
                    })
                    const url = URL.createObjectURL(blob)
                    const link = document.createElement('a')
                    link.href = url
                    link.download = `mini-binding-settings-${Date.now()}.json`
                    link.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="w-full border-blue-300 text-blue-700 hover:bg-blue-50"
                >
                  <Database className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {hasChanges && (
        <div className="fixed bottom-4 right-4 p-4 bg-card border rounded-lg shadow-lg">
          <p className="text-sm font-medium mb-2">You have unsaved changes</p>
          <div className="flex gap-2">
            <Button size="sm" onClick={saveSettings} disabled={isLoading}>
              Save Changes
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setSettings(JSON.parse(localStorage.getItem('mini-binding-settings') || JSON.stringify(DEFAULT_SETTINGS)))
                setHasChanges(false)
              }}
            >
              Discard
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
