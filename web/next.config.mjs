/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000',
    ENABLE_MOCK: process.env.ENABLE_MOCK || '0',
  },
  images: {
    domains: ['localhost'],
  },
}

export default nextConfig
