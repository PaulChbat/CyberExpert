/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true, // Or whatever options you already have

    // 4. CONFIGURE EXTERNAL IMAGE DOMAIN
    images: {
      remotePatterns: [
        {
          protocol: 'https', // Protocol used by the image URL
          hostname: 'exeo.net', // The domain hosting the image
          port: '', // Usually leave empty for default ports (80/443)
          pathname: '/wp-content/uploads/**', // Path prefix for images on that domain (be as specific as needed)
        },
        // Add other patterns here if needed for other domains
      ],
      // Alternatively, if you prefer the older 'domains' config (less specific):
      // domains: ['exeo.net'],
    },

};

export default nextConfig;
