import { Inter } from 'next/font/google'
import '../styles/globals.css'
import { Providers } from './providers'
import { Header } from '@/components/Header'
import { ProvenanceFooter } from '@/components/ProvenanceFooter'
import { MockModeBanner } from '@/components/MockModeBanner'
import { Toaster } from '@/components/ui/toaster'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Mini Binding - Ligand-Protein Binding Prediction',
  description: 'Scientist-grade UI for ligand-protein binding prediction with calibrated probabilities and explanations',
  keywords: ['drug discovery', 'binding prediction', 'machine learning', 'cheminformatics'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen flex flex-col bg-background">
            <MockModeBanner />
            <Header />
            <main className="flex-1 container mx-auto px-4 py-6">
              {children}
            </main>
            <ProvenanceFooter />
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  )
}
