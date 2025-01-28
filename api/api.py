import os, sys
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import uvicorn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
import numpy as np
from datetime import datetime
# Add the 'scripts' directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join('..', 'scripts')))
from feature_engg import FeatureEngineering # type: ignore
from credit_scoring_model import CreditScoreRFM # type: ignore

def load_model(model_path='model/best_model.pkl'):
    """Loads the model from local storage."""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = joblib.load(model_path)
    print(type(model))
    return model


class InputData(BaseModel):
    TransactionId: int
    CustomerId: int
    ProductCategory: str
    ChannelId: str
    Amount: float
    TransactionStartTime: datetime
    PricingStrategy: int

app = FastAPI()
model = load_model()


@app.post("/predict")
async def predict(input_data: InputData):
    """Predicts using the loaded model."""
    try:
        # Convert input data to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        # Initialize feature engineering and RFM classes
        feature_engineering = FeatureEngineering()
        credit_score_rfm = CreditScoreRFM(rfm_data=df.copy())

        # 1. Calculate RFM features
        rfm_data = credit_score_rfm.calculate_rfm()

        # 2. Calculate RFM Score and label
        rfm_data = credit_score_rfm.calculate_rfm_scores(rfm_data)

         # 3. Merge RFM Features back to the input dataframe
        df = pd.merge(df, rfm_data[['CustomerId','Recency', 'Frequency', 'Monetary','Risk_Label']], on='CustomerId', how='left')

        # 4. Preprocess Features
        df = feature_engineering.create_aggregate_features(df)
        df = feature_engineering.create_transaction_features(df)
        df = feature_engineering.extract_time_features(df)
        categorical_cols =['ProductCategory','ChannelId', 'PricingStrategy']
        df = feature_engineering.encode_categorical_features(df, categorical_cols)
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        df = feature_engineering.normalize_numerical_features(df, numerical_cols, method='standardize')

        # 5. Feature selection
        correlation_matrix = df.corr()
        corr_with_target = correlation_matrix['Risk_Label'].abs()
        selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
        selected_features.remove('Risk_Label')  # Exclude the target column itself
        X = df[selected_features]


        # Make prediction
        prediction = model.predict(X)[0]
        return JSONResponse(content={"prediction": int(prediction)})

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)