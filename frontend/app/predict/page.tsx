"use client";
import { useState } from "react";

export default function PredictPage() {
    const [features, setFeatures] = useState<string[]>(["0", "0", "0", "0", "0"]);
    const [prediction, setPrediction] = useState<number | null>(null);
    const [error, setError] = useState("");

    const handleInputChange = (index: number, value: string) => {
        const newFeatures = [...features];
        newFeatures[index] = value;
        setFeatures(newFeatures);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setPrediction(null);

        const numericFeatures = features.map(parseFloat);
        if (numericFeatures.some(isNaN)) {
            setError("All inputs must be valid numbers.");
            return;
        }

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: numericFeatures }),
            });
            const data = await res.json();
            if (res.ok) {
                setPrediction(data.prediction);
            } else {
                setError(data.detail);
            }
        } catch (err) {
            setError(String(err));
        }
    };

    return (
        <div>
            <h1>Predict</h1>
            <form onSubmit={handleSubmit} style={{ maxWidth: "400px" }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                    {features.map((val, idx) => (
                        <label key={idx}>
                            Feature {idx + 1}:
                            <input
                                type="number"
                                step="any"
                                value={val}
                                onChange={(e) => handleInputChange(idx, e.target.value)}
                                style={{ width: "100%", marginTop: "0.25rem" }}
                            />
                        </label>
                    ))}
                </div>
                <button type="submit" style={{ marginTop: "1rem", width: "100%" }}>Predict</button>
            </form>

            {error && <p style={{ color: "red", marginTop: "1rem" }}>Error: {error}</p>}

            {prediction !== null && (
                <div style={{ marginTop: "1rem", fontSize: "1.2rem", fontWeight: "bold" }}>
                    Prediction: {prediction.toFixed(6)}
                </div>
            )}
        </div>
    );
}
