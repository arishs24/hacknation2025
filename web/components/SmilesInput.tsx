"use client"

import * as React from "react"
import { Beaker, RotateCcw, Trash2, CheckCircle, AlertCircle, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { cn, validateSmiles } from "@/lib/format"

interface SmilesInputProps {
  value: string
  onChange: (value: string) => void
  onSanitize: () => void
  onStandardize: () => void
}

export function SmilesInput({ value, onChange, onSanitize, onStandardize }: SmilesInputProps) {
  const [validation, setValidation] = React.useState<ReturnType<typeof validateSmiles> | null>(null)

  // Validate SMILES on change
  React.useEffect(() => {
    if (value.trim()) {
      setValidation(validateSmiles(value.trim()))
    } else {
      setValidation(null)
    }
  }, [value])

  const handleClear = () => {
    onChange("")
    setValidation(null)
  }

  return (
    <div className="space-y-3">
      <Label htmlFor="smiles-input">SMILES String</Label>
      
      <div className="relative">
        <Textarea
          id="smiles-input"
          placeholder="Enter SMILES string (e.g., CC1=CC(=O)NC(C)=N1)..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={cn(
            "smiles-input min-h-[100px] resize-none",
            validation?.valid === false && "border-destructive focus-visible:ring-destructive"
          )}
          spellCheck={false}
        />
        
        {value && (
          <Button
            variant="ghost"
            size="sm"
            className="absolute top-2 right-2"
            onClick={handleClear}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={onSanitize}
          disabled={!value.trim()}
        >
          <Beaker className="h-4 w-4 mr-1" />
          Sanitize
        </Button>
        
        <Button
          variant="outline"
          size="sm"
          onClick={onStandardize}
          disabled={!value.trim()}
        >
          <RotateCcw className="h-4 w-4 mr-1" />
          Standardize
        </Button>
      </div>

      {/* Validation status and molecule info */}
      {validation && (
        <div className="space-y-2">
          <div className={cn(
            "flex items-center gap-2 text-sm",
            validation.valid ? "text-green-600" : "text-destructive"
          )}>
            {validation.valid ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            {validation.valid ? "Valid SMILES" : validation.message}
          </div>

          {validation.valid && (
            <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
              {validation.atomCount && (
                <div className="flex items-center gap-1">
                  <Info className="h-3 w-3" />
                  {validation.atomCount} atoms
                </div>
              )}
              {validation.molecularWeight && (
                <div>MW: {validation.molecularWeight}</div>
              )}
              {validation.logP && (
                <div>logP: {validation.logP}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Help text */}
      <div className="text-xs text-muted-foreground">
        Enter a valid SMILES string representing your molecule of interest. 
        Use Sanitize to clean the input or Standardize to normalize salts and stereochemistry.
      </div>
    </div>
  )
}
