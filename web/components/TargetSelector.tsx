"use client"

import * as React from "react"
import { Search, ChevronDown } from "lucide-react"
import { type Target } from "@/lib/schemas"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/format"

interface TargetSelectorProps {
  targets: Target[]
  value: Target | null
  onChange: (target: Target | null) => void
}

export function TargetSelector({ targets, value, onChange }: TargetSelectorProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const [searchQuery, setSearchQuery] = React.useState("")

  const filteredTargets = targets.filter(target =>
    target.target_entry.toLowerCase().includes(searchQuery.toLowerCase()) ||
    target.target_id.toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const targetSelector = document.getElementById('target-selector-container')
      if (targetSelector && !targetSelector.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const handleSelect = (target: Target) => {
    onChange(target)
    setIsOpen(false)
    setSearchQuery("")
  }

  return (
    <div className="space-y-2">
      <Label htmlFor="target-selector">Protein Target</Label>
      
      <div className="relative">
        <Button
          id="target-selector"
          variant="outline"
          className={cn(
            "w-full justify-between text-left font-normal",
            !value && "text-muted-foreground"
          )}
          onClick={() => setIsOpen(!isOpen)}
        >
          {value ? (
            <span className="flex items-center gap-2">
              <span className="font-mono text-xs bg-muted px-1 rounded">
                {value.target_id}
              </span>
              {value.target_entry}
            </span>
          ) : (
            "Select a protein target..."
          )}
          <ChevronDown className="h-4 w-4 opacity-50" />
        </Button>

        {isOpen && (
          <Card className="absolute top-full mt-1 w-full z-50 max-h-80 overflow-hidden">
            <CardContent className="p-0">
              <div className="p-3 border-b">
                <div className="relative">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search targets..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8"
                    autoFocus
                  />
                </div>
              </div>
              
              <div className="max-h-60 overflow-auto">
                {filteredTargets.length === 0 ? (
                  <div className="p-4 text-center text-muted-foreground">
                    No targets found
                  </div>
                ) : (
                  filteredTargets.map((target) => (
                    <button
                      key={target.target_id}
                      className="w-full text-left p-3 hover:bg-muted transition-colors border-b last:border-b-0"
                      onClick={() => handleSelect(target)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs bg-muted px-1 rounded">
                              {target.target_id}
                            </span>
                            <span className="font-medium">{target.target_entry}</span>
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            Train: {target.n_train} | Val: {target.n_val} | Test: {target.n_test}
                          </div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {(target.pos_frac_train * 100).toFixed(0)}% pos
                        </div>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {value && (
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Training samples: {value.n_train} ({(value.pos_frac_train * 100).toFixed(1)}% positive)</div>
          <div>Validation: {value.n_val} • Test: {value.n_test}</div>
        </div>
      )}
    </div>
  )
}
