"use client";
import { useState } from "react";

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState("");
    const [loading, setLoading] = useState(false);
    const [stats, setStats] = useState<any>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setStats(null); // Clear previous stats
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            let data;
            const contentType = res.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                data = await res.json();
            } else {
                const text = await res.text();
                throw new Error(`Server returned non-JSON response: ${text.slice(0, 100)}...`);
            }

            if (res.ok) {
                setStatus(`✓ Success: ${data.filename} uploaded`);
                setStats(data.statistics);
            } else {
                setStatus(`✗ Error: ${data.detail || JSON.stringify(data)}`);
                setStats(null);
            }
        } catch (err) {
            setStatus(`✗ Error: ${err}`);
            setStats(null);
        }
        setLoading(false);
    };

    return (
        <div>
            <h1>Upload Dataset</h1>
            <p>Upload a .pkl file containing 5D dataset (X: 5 features, y: target).</p>
            <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
                <input type="file" onChange={handleFileChange} accept=".pkl" />
                <button type="submit" disabled={!file || loading} style={{ marginLeft: "1rem" }}>
                    {loading ? "Uploading..." : "Upload"}
                </button>
            </form>
            {status && <p style={{ marginTop: "1rem", fontWeight: "bold" }}>{status}</p>}

            {stats && (
                <div style={{ marginTop: "2rem", border: "2px solid #4CAF50", padding: "1.5rem", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                    <h2 style={{ marginTop: 0 }}>Dataset Statistics</h2>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                        <div>
                            <h3>Overview</h3>
                            <ul style={{ listStyle: "none", padding: 0 }}>
                                <li><strong>Total Samples:</strong> {stats.total_samples.toLocaleString()}</li>
                                <li><strong>Features:</strong> {stats.n_features}</li>
                                <li><strong>Missing Values (X):</strong> {stats.missing_values.X}</li>
                                <li><strong>Missing Values (y):</strong> {stats.missing_values.y}</li>
                            </ul>
                        </div>

                        <div>
                            <h3>Target Distribution</h3>
                            <ul style={{ listStyle: "none", padding: 0 }}>
                                <li><strong>Min:</strong> {stats.target_distribution.min.toFixed(4)}</li>
                                <li><strong>Max:</strong> {stats.target_distribution.max.toFixed(4)}</li>
                                <li><strong>Mean:</strong> {stats.target_distribution.mean.toFixed(4)}</li>
                            </ul>
                        </div>
                    </div>

                    <div style={{ marginTop: "1rem" }}>
                        <h3>Planned Train/Val/Test Split</h3>
                        <div style={{ display: "flex", gap: "2rem" }}>
                            <div>
                                <strong>Train:</strong> {stats.planned_split.train.toLocaleString()}
                                <span style={{ color: "#666", marginLeft: "0.5rem" }}>
                                    ({(stats.planned_split.train / stats.total_samples * 100).toFixed(1)}%)
                                </span>
                            </div>
                            <div>
                                <strong>Validation:</strong> {stats.planned_split.validation.toLocaleString()}
                                <span style={{ color: "#666", marginLeft: "0.5rem" }}>
                                    ({(stats.planned_split.validation / stats.total_samples * 100).toFixed(1)}%)
                                </span>
                            </div>
                            <div>
                                <strong>Test:</strong> {stats.planned_split.test.toLocaleString()}
                                <span style={{ color: "#666", marginLeft: "0.5rem" }}>
                                    ({(stats.planned_split.test / stats.total_samples * 100).toFixed(1)}%)
                                </span>
                            </div>
                        </div>
                    </div>

                    {stats.missing_values.X === 0 && stats.missing_values.y === 0 && (
                        <p style={{ marginTop: "1rem", color: "#4CAF50", fontWeight: "bold" }}>
                            ✓ Data validation passed: No missing values detected
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}
