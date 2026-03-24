import type { Metadata } from "next";
import { Rajdhani } from "next/font/google";
import "./globals.css";

const rajdhani = Rajdhani({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-rajdhani",
});

export const metadata: Metadata = {
  title: "Athlink CV — Professional Football Analysis for Everyone",
  description:
    "Upload a football clip, get a full AI coaching report in 90 seconds. Player tracking, team analysis, tactical insights. £30/month — not £10,000.",
  keywords: [
    "football analysis",
    "AI coaching",
    "player tracking",
    "tactical analysis",
    "sports analytics",
    "football video analysis",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={rajdhani.variable}>
      <body className={`${rajdhani.className} antialiased`}>
        <div className="noise-overlay" />
        {children}
      </body>
    </html>
  );
}
