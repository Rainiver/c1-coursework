from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import os
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from backend modules
# Import from backend package
from interpolator.data import DataHandler
from interpolator.model import ModelHandler

app = FastAPI(title="Interpolator API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify generic specifics
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_handler = DataHandler()
model_handler = ModelHandler()

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class TrainConfig(BaseModel):
    hidden_layers: list[int] = [64, 32, 16]
    learning_rate: float = 0.001
    max_epochs: int = 100

class PredictInput(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename
    if not filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .pkl is supported.")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        X, y = data_handler.load_dataset(file_path)
        
        # Calculate detailed statistics
        total_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Check for missing values
        missing_X = int(np.sum(np.isnan(X)))
        missing_y = int(np.sum(np.isnan(y)))
        
        # Get data ranges
        y_min, y_max, y_mean = float(np.min(y)), float(np.max(y)), float(np.mean(y))
        
        # Simulate split sizes (actual split happens during training)
        train_size = int(total_samples * 0.6)
        val_size = int(total_samples * 0.2)
        test_size = total_samples - train_size - val_size
        
        return {
            "filename": filename,
            "status": "uploaded",
            "statistics": {
                "total_samples": total_samples,
                "n_features": n_features,
                "missing_values": {
                    "X": missing_X,
                    "y": missing_y
                },
                "target_distribution": {
                    "min": y_min,
                    "max": y_max,
                    "mean": y_mean
                },
                "planned_split": {
                    "train": train_size,
                    "validation": val_size,
                    "test": test_size
                }
            }
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")


@app.post("/train")
def train_model(config: TrainConfig):
    if data_handler.X is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload first.")
    
    try:
        # Initialize model with config (using optimal hyperparameters from tuning)
        global model_handler
        model_handler = ModelHandler(
            hidden_layers=config.hidden_layers if config.hidden_layers else [256, 128, 64, 32],
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs
        )

        
        # Split into train/val/test (70/10/20)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.preprocess_and_split()
        
        # Train with validation for early stopping (verbose=True for terminal output)
        print("\n" + "="*60)
        print(f"Starting training with {len(X_train)} samples...")
        print("="*60)
        results = model_handler.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)

        
        # Evaluate on test set
        preds_test = model_handler.predict(X_test)
        test_mse = np.mean((preds_test - y_test)**2)
        test_rmse = np.sqrt(test_mse)
        r2 = 1 - (np.sum((y_test - preds_test)**2) / np.sum((y_test - np.mean(y_test))**2))
        
        # Print summary to terminal
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Test RMSE:        {test_rmse:.6f}")
        print(f"Test R² Score:    {r2:.4f}")
        print(f"Training Time:    {results['training_time']:.2f}s")
        print(f"Epochs Trained:   {len(results['losses'])}")
        print(f"Training Samples: {len(X_train)}")
        print(f"Validation Samples: {len(X_val)}")
        print(f"Test Samples:     {len(X_test)}")
        print("="*60 + "\n")
        
        return {
            "status": "training_complete",
            "metrics": {
                "training_time": float(results["training_time"]),
                "final_loss": float(results["losses"][-1]) if results["losses"] else 0.0,
                "test_mse": float(test_mse),
                "test_rmse": float(test_rmse),
                "test_r2": float(r2),
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "epochs_trained": len(results["losses"])
            }
        }
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/predict")
def predict(input_data: PredictInput):
    if model_handler.model is None:
        raise HTTPException(status_code=400, detail="Model not trained.")
    
    if len(input_data.features) != 5:
         raise HTTPException(status_code=400, detail="Input must have 5 features.")
         
    try:
        # Scale input using the scaler fitted on training data
        input_array = np.array(input_data.features).reshape(1, -1)
        scaled_input = data_handler.scaler.transform(input_array)
        
        prediction = model_handler.predict(scaled_input)
        return {"prediction": float(prediction[0])}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
