"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Beaker, BarChart3, Database, Settings, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/format"

const navigation = [
  { name: "Predict", href: "/predict", icon: Beaker },
  { name: "Batch", href: "/batch", icon: Upload },
  { name: "Metrics", href: "/metrics", icon: BarChart3 },
  { name: "Data", href: "/data", icon: Database },
  { name: "Settings", href: "/settings", icon: Settings },
]

export function Header() {
  const pathname = usePathname()

  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <Beaker className="h-6 w-6 text-primary" />
            <span className="font-bold text-xl">Mini Binding</span>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              
              return (
                <Button
                  key={item.name}
                  variant={isActive ? "default" : "ghost"}
                  size="sm"
                  asChild
                >
                  <Link href={item.href} className="flex items-center gap-2">
                    <Icon className="h-4 w-4" />
                    {item.name}
                  </Link>
                </Button>
              )
            })}
          </nav>
        </div>
      </div>
    </header>
  )
}
