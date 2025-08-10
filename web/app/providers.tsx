'use client'

import { useEffect } from 'react'
import { useAppStore } from '@/lib/store'

export function Providers({ children }: { children: React.ReactNode }) {
  const { mockMode, darkMode, setMockMode } = useAppStore()

  useEffect(() => {
    // Mock mode is disabled - using real API
    setMockMode(false)

    // Apply dark mode class
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [mockMode, darkMode, setMockMode])

  return <>{children}</>
}
