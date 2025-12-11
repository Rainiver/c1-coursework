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
        {children}
      </body>
    </html>
  );
}
