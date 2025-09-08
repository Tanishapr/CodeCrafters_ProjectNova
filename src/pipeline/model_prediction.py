"""
Module: model_prediction.py
Author: Tanisha Priya, Harshita Jangde, Prachi Tavse
Date: 2025-09-08
Description:
    Handles preprocessing of user input and centralized prediction of Nova Credit Score
    using the trained XGBoost pipeline saved in 'models/credit_model.pkl'.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained pipeline (includes scaler and model)
pipeline = joblib.load("models/credit_model.pkl")

def preprocess_user_input(user_data: dict) -> pd.DataFrame:
    """
    Preprocesses a single user input dictionary into a DataFrame suitable for prediction.

    Parameters:
        user_data (dict): Raw input features from the user.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for pipeline prediction.
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([user_data])

    # One-hot encode categorical features
    categorical_cols = ['zone', 'job_type', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Scale numeric features
    numeric_cols = [
        'earnings_weekly', 'trip_count_monthly', 'avg_rating_last_90_days',
        'cancellation_rate', 'age', 'rent_paid_last_12_months',
        'utility_paid_last_12', 'app_logins_last_30_days', 
        'complaints_last_6_months', 'outstanding_loans'
    ]
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    return df_encoded

def predict_credit_score(user_data: dict) -> float:
    """
    Centralized function to preprocess user input and predict Nova Credit Score.

    Parameters:
        user_data (dict): Raw input features from the user.

    Returns:
        float: Predicted credit score.
    """
    df_processed = preprocess_user_input(user_data)
    prediction = pipeline.predict(df_processed)[0]
    return float(prediction)