import * as React from "react"
import { ExternalLink } from "lucide-react"

export function ProvenanceFooter() {
  return (
    <footer className="border-t bg-muted/30 text-xs text-muted-foreground">
      <div className="container mx-auto px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-3">
            <a 
              href="https://www.bindingdb.org" 
              className="flex items-center gap-1 hover:text-foreground transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              BindingDB Articles
              <ExternalLink className="h-3 w-3" />
            </a>
            <span>•</span>
            <span>Label pX≥7 / ≤5</span>
            <span>•</span>
            <span>Split: hash (scaffold in repo)</span>
            <span>•</span>
            <span>Commit abc123</span>
            <span>•</span>
            <span>Seed 42</span>
          </div>
          
          <div className="text-right">
            <div>Screening-only • Not medical advice</div>
          </div>
        </div>
      </div>
    </footer>
  )
}
