from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import date
import uvicorn

app = FastAPI()

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
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

    # Ensure all expected dummy variables are present
    expected_season_cols = ['Season_Fall', 'Season_Spring', 'Season_Summer']
    for col in expected_season_cols:
        if col not in df.columns:
            df[col] = 0

    # Time features
    df['Year'] = df['ds'].dt.year
    df['Month'] = df['ds'].dt.month
    df['WeekOfYear'] = df['ds'].dt.isocalendar().week.astype(int)

    # Drop date column before prediction
    return df.drop(columns=['ds'])

# === Load Trained Model ===
def load_model(category: str):
    try:
        model = joblib.load(f"Model/rf_model_{category}.joblib")
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model for category '{category}' not found.")

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

        # Make predictions
        predictions = model.predict(X)
        df['prediction'] = predictions  # Use 'prediction' instead of 'yhat'

        return df[['ds', 'prediction']].to_dict(orient='records')

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# === Run Locally ===
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
