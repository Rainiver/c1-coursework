"use client";
import { useState } from "react";

export default function TrainPage() {
    const [config, setConfig] = useState({
        hidden_layers: "64,32,16",
        learning_rate: 0.001,
        max_epochs: 100,
    });
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setResult(null);

        const payload = {
            hidden_layers: config.hidden_layers.split(",").map((x) => parseInt(x.trim())),
            learning_rate: config.learning_rate,
            max_epochs: config.max_epochs,
        };

        try {
            const res = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (res.ok) {
                setResult(data);
            } else {
                setError(data.detail);
            }
        } catch (err) {
            setError(String(err));
        }
        setLoading(false);
    };

    return (
        <div>
            <h1>Train Model</h1>
            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1rem", maxWidth: "400px" }}>
                <label>
                    Hidden Layers (comma separated):
                    <br />
                    <input
                        type="text"
                        value={config.hidden_layers}
                        onChange={(e) => setConfig({ ...config, hidden_layers: e.target.value })}
                        style={{ width: "100%" }}
                    />
                </label>
                <label>
                    Learning Rate:
                    <br />
                    <input
                        type="number"
                        step="0.0001"
                        value={config.learning_rate}
                        onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                        style={{ width: "100%" }}
                    />
                </label>
                <label>
                    Max Epochs:
                    <br />
                    <input
                        type="number"
                        value={config.max_epochs}
                        onChange={(e) => setConfig({ ...config, max_epochs: parseInt(e.target.value) })}
                        style={{ width: "100%" }}
                    />
                </label>
                <button type="submit" disabled={loading}>
                    {loading ? "Training..." : "Start Training"}
                </button>
            </form>

            {error && <p style={{ color: "red", marginTop: "1rem" }}>Error: {error}</p>}

            {result && (
                <div style={{ marginTop: "1rem", border: "1px solid #ccc", padding: "1rem" }}>
                    <h3>Training Results</h3>
                    <p>Status: {result.status}</p>
                    <ul>
                        <li>Training Time: {result.metrics.training_time.toFixed(4)} s</li>
                        <li>Final Loss: {result.metrics.final_loss.toFixed(6)}</li>
                        <li>Validation MSE: {result.metrics.val_mse.toFixed(6)}</li>
                        <li>Validation R²: {result.metrics.val_r2.toFixed(4)}</li>
                    </ul>
                </div>
            )}
        </div>
    );
}
