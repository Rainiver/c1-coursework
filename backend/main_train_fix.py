@app.post("/train")
@app.post("/api/train")
def train_model(config: TrainConfig):
    if data_handler.X is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload first.")
    
    try:
        # Initialize model with config
        global model_handler
        model_handler = ModelHandler(
            hidden_layers=config.hidden_layers if config.hidden_layers else [128, 64, 32],
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs
        )
        
        # Split and preprocess
        (X_train, y_train), (X_val, y_val), _ = data_handler.preprocess_and_split()
        
        # Train with validation set
        results = model_handler.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Evaluate on validation
        preds_val = model_handler.predict(X_val)
        val_mse = float(np.mean((preds_val - y_val) ** 2))
        val_r2 = 1 - (np.sum((y_val - preds_val)**2) / np.sum((y_val - np.mean(y_val))**2))
        
        return {
            "status": "training_complete",
            "metrics": {
                "training_time": float(results["training_time"]),
                "final_loss": float(results["losses"][-1]) if results["losses"] else 0.0,
                "val_mse": val_mse,
                "val_r2": float(val_r2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
