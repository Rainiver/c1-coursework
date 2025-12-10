import Link from "next/link";
import "./globals.css";

export const metadata = {
  title: "Fivedreg Interpolator",
  description: "Learn and serve 5D numerical datasets",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav style={{ padding: "1rem", backgroundColor: "#333", color: "white", marginBottom: "2rem" }}>
          <ul style={{ display: "flex", gap: "2rem", listStyle: "none", margin: 0, padding: 0 }}>
            <li><Link href="/" style={{ color: "white", textDecoration: "none", fontWeight: "bold" }}>Home</Link></li>
            <li><Link href="/upload" style={{ color: "white", textDecoration: "none" }}>Upload</Link></li>
            <li><Link href="/train" style={{ color: "white", textDecoration: "none" }}>Train</Link></li>
            <li><Link href="/predict" style={{ color: "white", textDecoration: "none" }}>Predict</Link></li>
          </ul>
        </nav>
        <main style={{ padding: "0 2rem", maxWidth: "1200px", margin: "0 auto" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
