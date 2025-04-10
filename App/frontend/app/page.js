// File: app/page.js
// NO "use client" directive here! This is now a Server Component.

import HomePageClient from './HomePageClient'; // Import the Client Component

// Export metadata - This is allowed in Server Components
export const metadata = {
  title: 'EXEO CyberExpert | Log Analyzer', // Set your desired page title here
  description: 'Analyze security logs to generate summaries and mitigation actions.',
  icons: {
    icon: 'https://exeo.net/wp-content/uploads/2023/02/cropped-Exeo-Icon-512x512-Transparent-32x32.png', // Make sure favicon.ico is in the app/ directory
  },
};

// This is the main Page component (Server Component)
export default function HomePage() {
  // It simply renders the Client Component which contains all the logic
  return <HomePageClient />;
}