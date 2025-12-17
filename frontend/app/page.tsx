"use client";
import { useState, useRef } from "react";
import "./apple-style.css";

// API URL: always use localhost:8000 from browser (Docker port mapping)
const API_URL = 'http://localhost:8000';


export default function InterpolatorPage() {
  const [activeStep, setActiveStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [datasetStats, setDatasetStats] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState("");
  const [trainingEpochProgress, setTrainingEpochProgress] = useState(0); // 0-100%
  const [epochs, setEpochs] = useState(200);
  const [trainedModel, setTrainedModel] = useState(false);
  const [modelMetrics, setModelMetrics] = useState<any>(null);
  const [predictionInputs, setPredictionInputs] = useState([0.5, 0.5, 0.5, 0.5, 0.5]);
  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Step 1: Upload
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setUploadStatus("success");
        setDatasetStats(data.statistics);
        setActiveStep(2);
      } else {
        setUploadStatus("error");
      }
    } catch (err) {
      setUploadStatus("error");
    }
  };

  // Step 2: Train
  const handleTrain = async () => {
    if (!file) return;
    setTraining(true);
    setTrainingProgress("Initializing training...");
    setTrainingEpochProgress(0);

    // Smooth progress animation
    const progressInterval = setInterval(() => {
      setTrainingEpochProgress((prev) => Math.min(prev + 1, 95));
    }, (epochs * 20) / 100);

    try {
      const res = await fetch(`${API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hidden_layers: [256, 128, 64, 32],  // Optimal configuration from hyperparameter tuning
          learning_rate: 0.001,
          max_epochs: epochs,
        }),
      });


      clearInterval(progressInterval);
      setTrainingEpochProgress(100);

      const data = await res.json();

      if (res.ok) {
        setModelMetrics(data.metrics);
        setTrainingProgress("Training complete!");
        setTrainedModel(true);
        setTimeout(() => setActiveStep(3), 1000);
      }
    } catch (err) {
      clearInterval(progressInterval);
      setTrainingProgress("Training failed");
      console.error("Training error:", err);
    } finally {
      setTraining(false);
    }
  };

  // Step 3: Predict
  const handlePredict = async () => {
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: predictionInputs }),
      });

      const data = await res.json();
      if (res.ok) {
        setPredictionResult(data.prediction);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const randomizeInputs = () => {
    setPredictionInputs(Array(5).fill(0).map(() => Math.random()));
  };

  const resetInputs = () => {
    setPredictionInputs([0.5, 0.5, 0.5, 0.5, 0.5]);
  };

  const handleInputChange = (index: number, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 1) {
      const newInputs = [...predictionInputs];
      newInputs[index] = numValue;
      setPredictionInputs(newInputs);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <div className="icon">🧮</div>
        <h1>5D Neural Network Interpolator</h1>
        <p className="subtitle">Upload your 5D dataset, train a neural network model, and test predictions</p>
        <div className="backend-status">
          <span className="status-dot"></span> Backend Connected
        </div>
      </header>

      {/* Step Indicators */}
      <div className="steps">
        <div className={`step ${activeStep >= 1 ? "active" : ""} ${uploadStatus === "success" ? "complete" : ""}`}>
          <div className="step-number">
            {uploadStatus === "success" ? "✓" : "1"}
          </div>
          <div className="step-info">
            <div className="step-title">Upload Data</div>
            <div className="step-desc">Upload your .pkl dataset</div>
          </div>
        </div>

        <div className={`step ${activeStep >= 2 ? "active" : ""} ${trainedModel ? "complete" : ""}`}>
          <div className="step-number">
            {trainedModel ? "✓" : "2"}
          </div>
          <div className="step-info">
            <div className="step-title">Train Model</div>
            <div className="step-desc">Train neural network</div>
          </div>
        </div>

        <div className={`step ${activeStep >= 3 ? "active" : ""}`}>
          <div className="step-number">3</div>
          <div className="step-info">
            <div className="step-title">Test Predictions</div>
            <div className="step-desc">Test model predictions</div>
          </div>
        </div>
      </div>

      {/* Step 1: Upload */}
      {activeStep === 1 && (
        <div className="card">
          <h2>Step 1: Upload Your Dataset</h2>

          <div
            className="upload-zone"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              {file ? file.name : "Click to upload or drag and drop"}
            </p>
            <p className="upload-hint">.pkl files only</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pkl"
              onChange={handleFileSelect}
              style={{ display: "none" }}
            />
          </div>

          {file && (
            <button className="btn-primary" onClick={handleUpload}>
              Upload Dataset
            </button>
          )}
        </div>
      )}

      {/* Step 2: Train */}
      {activeStep === 2 && (
        <div className="card">
          {uploadStatus === "success" && (
            <div className="success-banner">
              ✓ Successfully uploaded {datasetStats?.total_samples.toLocaleString()} data points!
            </div>
          )}

          <h2>Step 2: Train Your Model</h2>

          {datasetStats && (
            <div className="dataset-info">
              <h3>Dataset Information</h3>
              <ul>
                <li>• {datasetStats.total_samples.toLocaleString()} data points</li>
                <li>• {datasetStats.n_features} input features</li>
                <li>• Target range: [{datasetStats.target_distribution.min.toFixed(3)}, {datasetStats.target_distribution.max.toFixed(3)}]</li>
              </ul>
            </div>
          )}

          <div className="train-section">
            <h3>Train Model</h3>

            <div className="input-group">
              <label>Training Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                min="10"
                max="1000"
                disabled={training}
              />
              <small>More epochs = better accuracy, longer training time</small>
            </div>

            <button
              className="btn-primary"
              onClick={handleTrain}
              disabled={training}
            >
              {training ? "Training..." : "Start Training"}
            </button>

            {training && (
              <div className="training-progress">
                <h4>Training Progress</h4>
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${trainingEpochProgress}%` }}></div>
                </div>
                <div className="progress-text">
                  {trainingEpochProgress < 100 ? `Training: ${trainingEpochProgress}%` : "Finalizing..."}
                </div>
              </div>
            )}

            {trainingProgress && !training && (
              <div className="training-result">
                <p>{trainingProgress}</p>
              </div>
            )}
          </div>

          <div className="button-row">
            {trainedModel && (
              <button className="btn-primary" onClick={() => setActiveStep(3)}>
                Continue to Predictions →
              </button>
            )}
            <button className="btn-text" onClick={() => { setActiveStep(1); setTrainedModel(false); setUploadStatus(""); }}>
              ← Start Over
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Predict */}
      {activeStep === 3 && (
        <div className="card">
          {modelMetrics && (
            <div className="model-performance">
              <h3>Model Performance</h3>
              <div className="metrics-grid">
                <div className="metric">
                  <div className="metric-label">Test RMSE</div>
                  <div className="metric-value">{modelMetrics.test_rmse?.toFixed(6) || 'N/A'}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">R² Score</div>
                  <div className="metric-value">{modelMetrics.test_r2?.toFixed(4) || 'N/A'}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">Training Time</div>
                  <div className="metric-value">{modelMetrics.training_time?.toFixed(2) || 'N/A'}s</div>
                </div>
              </div>
            </div>
          )}



          <h2>Test Prediction</h2>

          <div className="sliders">
            {predictionInputs.map((value, idx) => (
              <div key={idx} className="slider-group">
                <label>X{idx + 1}</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={value}
                    onChange={(e) => {
                      const newInputs = [...predictionInputs];
                      newInputs[idx] = parseFloat(e.target.value);
                      setPredictionInputs(newInputs);
                    }}
                  />
                  <input
                    type="number"
                    className="number-input"
                    min="0"
                    max="1"
                    step="0.01"
                    value={value.toFixed(2)}
                    onChange={(e) => handleInputChange(idx, e.target.value)}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="button-group">
            <button className="btn-secondary" onClick={randomizeInputs}>
              Randomize
            </button>
            <button className="btn-secondary" onClick={resetInputs}>
              Reset
            </button>
          </div>

          <button className="btn-primary" onClick={handlePredict}>
            Predict
          </button>

          {
            predictionResult !== null && (
              <div className="prediction-result">
                <h3>Prediction Result</h3>
                <div className="result-value">{predictionResult.toFixed(6)}</div>
                <div className="input-vector">
                  Input: [{predictionInputs.map(v => v.toFixed(2)).join(", ")}]
                </div>
              </div>
            )
          }

          <button className="btn-text" onClick={() => { setActiveStep(1); setTrainedModel(false); setUploadStatus(""); setPredictionResult(null); }}>
            ← Start Over
          </button>
        </div >
      )}
    </div >
  );
}
