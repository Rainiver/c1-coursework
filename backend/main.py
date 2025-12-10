from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fivedreg.data import DataHandler
from fivedreg.model import ModelHandler
import shutil
import os
import numpy as np

app = FastAPI(title="Fivedreg API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify generic specifics
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in a real app, might use dependency injection or database)
data_handler = DataHandler()
model_handler = ModelHandler()
# In Vercel (serverless), only /tmp is writable
UPLOAD_DIR = "/tmp"
# os.makedirs(UPLOAD_DIR, exist_ok=True) # /tmp always exists

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
    if not (filename.endswith(".pkl") or filename.endswith(".npz")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .pkl and .npz are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        data_handler.load_dataset(file_path)
        # Preprocess immediately or wait for train? Let's do a preliminary check.
        # But we don't split yet until training maybe? 
        # Requirement says "upload page... including validation and preview". 
        # So lets just validate load.
        return {"filename": filename, "status": "uploaded", "samples": data_handler.X.shape[0]}
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

@app.post("/train")
def train_model(config: TrainConfig):
    if data_handler.X is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload first.")
    
    try:
        # Update model config
        global model_handler
        model_handler = ModelHandler(
            hidden_layers=config.hidden_layers, 
            learning_rate=config.learning_rate, 
            max_epochs=config.max_epochs
        )
        
        # Split and preprocess
        (X_train, y_train), (X_val, y_val), _ = data_handler.preprocess_and_split()
        
        # Train
        results = model_handler.fit(X_train, y_train)
        
        # Evaluate on validation
        preds_val = model_handler.predict(X_val)
        mse = np.mean((preds_val - y_val)**2)
        r2 = 1 - (np.sum((y_val - preds_val)**2) / np.sum((y_val - np.mean(y_val))**2))
        
        return {
            "status": "training_complete",
            "metrics": {
                "training_time": results["training_time"],
                "final_loss": results["losses"][-1],
                "val_mse": float(mse),
                "val_r2": float(r2)
            }
        }
    except Exception as e:
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
