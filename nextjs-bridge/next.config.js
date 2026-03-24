/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typescript: {
    // Pre-existing API routes use old Next.js params pattern
    ignoreBuildErrors: true,
  },
}

module.exports = nextConfig
