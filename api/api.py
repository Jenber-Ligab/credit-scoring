from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime  # Import this for `datetime` type
import pickle
import sys, os
import numpy as np
import pandas as pd
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.feature_engg import FeatureEngineering  # Import the feature engineering class
from scripts.credit_scoring_model import CreditScoreRFM  # Import the RFM class

# Load the model
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    TransactionId: int
    CustomerId: int
    ProductCategory: int
    ChannelId: str
    Amount: float
    TransactionStartTime: datetime
    PricingStrategy: int

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Prepare input data as a DataFrame
        input_data_dict = {
            'TransactionId': input_data.TransactionId,
            'CustomerId': input_data.CustomerId,
            'ProductCategory': input_data.ProductCategory,
            'ChannelId': input_data.ChannelId,
            'Amount': input_data.Amount,
            'TransactionStartTime': input_data.TransactionStartTime,
            'PricingStrategy': input_data.PricingStrategy
        }

        input_df = pd.DataFrame([input_data_dict])

        # Feature Engineering
        fe = FeatureEngineering()
        
        # Creating aggregate and transaction features
        input_df = fe.create_aggregate_features(input_df)
        input_df = fe.create_transaction_features(input_df)
        
        # Extract time-based features
        input_df = fe.extract_time_features(input_df)

        # Encode categorical features
        categorical_cols = ['ProductCategory', 'ChannelId']
        input_df = fe.encode_categorical_features(input_df, categorical_cols)

        # Handle missing values and normalize numerical features
        numeric_cols = input_df.select_dtypes(include='number').columns.tolist()
        exclude_cols = ['Amount', 'TransactionId']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        input_df = fe.normalize_numerical_features(input_df, numeric_cols, method='standardize')

        # RFM Calculation for customer segmentation
        rfm = CreditScoreRFM(input_df.reset_index())
        rfm_df = rfm.calculate_rfm()
        
        # Merge the RFM features with the input data
        final_df = pd.merge(input_df, rfm_df, on='CustomerId', how='left')

        # Define all final features expected in the output
        final_features = [
            'PricingStrategy', 'Transaction_Count', 'Debit_Count', 'Credit_Count',
            'Debit_Credit_Ratio', 'Transaction_Month', 'Transaction_Year',
            'ProductCategory_financial_services', 'ChannelId_ChannelId_2',
            'ChannelId_ChannelId_3', 'Recency', 'Frequency'
        ]

        # Ensure all final features exist in the DataFrame and fill missing ones with 0
        final_df = final_df.reindex(columns=final_features, fill_value=0)

        # Make prediction
        prediction = model.predict(final_df)
        predicted_risk = 'Good' if prediction[0] == 0 else 'Bad'

        # Return the prediction result as a JSON response
        return {"customer_id": input_data.CustomerId, "predicted_risk": predicted_risk}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}