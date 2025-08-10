"use client"

import * as React from "react"
import { AlertTriangle, X } from "lucide-react"
import { useAppStore } from "@/lib/store"
import { Button } from "@/components/ui/button"

export function MockModeBanner() {
  const { mockMode, setMockMode } = useAppStore()
  const [isDismissed, setIsDismissed] = React.useState(false)

  if (!mockMode || isDismissed) {
    return null
  }

  return (
    <div className="bg-amber-50 border-b border-amber-200 text-amber-800 px-4 py-2">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium">
          <AlertTriangle className="h-4 w-4" />
          <span>Mock Mode Active</span>
          <span className="text-amber-600">•</span>
          <span className="text-amber-600">
            Using simulated data - backend unavailable
          </span>
        </div>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsDismissed(true)}
          className="h-6 w-6 p-0 text-amber-600 hover:text-amber-800"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
