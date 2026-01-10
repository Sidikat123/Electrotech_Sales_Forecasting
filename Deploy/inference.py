from fastapi import FastAPI, HTTPException
from pydantic import BaseModel                                                    
import pandas as pd
import joblib
import numpy as np
import os
import json
import traceback
from datetime import date
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title = 'Electrotech Sales Forecasting API', version = '1.0')

# === Request Schema ===
class ForecastRequest(BaseModel):
    category: str
    start_date: date
    end_date: date
    price: float
    season: str  # Expected: 'Winter', 'Spring', 'Summer', 'Fall'

# === Feature Engineering ===
def generate_features(df: pd.DataFrame, price: float, season: str) -> pd.DataFrame:
    df['Price'] = price
    df['Season'] = season

    # One-hot encode season
    df = pd.get_dummies(df, columns=['Season'])

    # Ensure all expected dummy variables are present
    expected_season_cols = ['Season_Spring', 'Season_Summer', 'Season_Winter']

    for col in expected_season_cols:
        if col not in df.columns:
            df[col] = 0

    # Time features
    df['Year'] = df['ds'].dt.year
    df['Month'] = df['ds'].dt.month
    df['WeekOfYear'] = df['ds'].dt.isocalendar().week.astype(int)
    df['Day'] = df['ds'].dt.day

    # Drop date column before prediction
    return df.drop(columns=['ds'])

# === Load Trained Model ===
ALLOWED_CATEGORIES = ["Accessories", "Laptop", "Smartphone", "Tablet"]

def load_model(category: str):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: '{category}'. Choose from {ALLOWED_CATEGORIES}")

    model_path = os.path.join("..", "Model", f"rf_model_{category}.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found at path: {model_path}")

    return joblib.load(model_path)

# === Prediction Endpoint ===
@app.post("/predict")
def predict(request: ForecastRequest):
    try:
        # Generate monthly timestamps
        dates = pd.date_range(start=request.start_date, end=request.end_date, freq='MS')
        if len(dates) == 0:
            raise HTTPException(status_code=400, detail="Date range must include at least one month.")

        # Create base DataFrame
        df = pd.DataFrame({'ds': dates})

        # Generate features for prediction
        X = generate_features(df.copy(), price=request.price, season=request.season)

        # Load the model
        model = load_model(request.category)

        # Load the exact training features
        feature_file = os.path.join("..", "Model", f"rf_model_{request.category}_features.json")
        if not os.path.exists(feature_file):
            raise HTTPException(status_code=500, detail=f"Feature file not found: {feature_file}")
        
        with open(feature_file, 'r') as f:
            feature_names = json.load(f)
        
        # Reindex to ensure all training features are present
        X = X.reindex(columns=feature_names, fill_value=0)

        # Make predictions
        predictions = model.predict(X)
        df['prediction'] = predictions  

        return df[['ds', 'prediction']].to_dict(orient='records')

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# === Run Locally ===
if __name__ == "__main__":
    print(f"Server is on port {os.getenv('port', 3000)}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv('port', 3000)))

