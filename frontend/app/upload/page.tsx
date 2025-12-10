"use client";
import { useState } from "react";

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
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
                setStatus(`Success: ${data.filename} uploaded. Total samples: ${data.samples}`);
            } else {
                setStatus(`Error: ${data.detail || JSON.stringify(data)}`);
            }
        } catch (err) {
            setStatus(`Error: ${err}`);
        }
        setLoading(false);
    };

    return (
        <div>
            <h1>Upload Dataset</h1>
            <p>Upload a .pkl or .npz file containing 5D dataset.</p>
            <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
                <input type="file" onChange={handleFileChange} accept=".pkl,.npz" />
                <button type="submit" disabled={!file || loading} style={{ marginLeft: "1rem" }}>
                    {loading ? "Uploading..." : "Upload"}
                </button>
            </form>
            {status && <p style={{ marginTop: "1rem" }}>{status}</p>}
        </div>
    );
}
