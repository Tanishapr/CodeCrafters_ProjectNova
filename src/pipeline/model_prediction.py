import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved pipeline (includes scaler)
pipeline = joblib.load("models/credit_model.pkl")

def preprocess_user_input(user_data: dict):
    """
    Convert user input dict into a DataFrame.
    The pipeline will handle scaling internally.
    """
    df = pd.DataFrame([user_data])
    categorical_cols = ['zone', 'job_type', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Scale numeric features
    numeric_cols = ['earnings_weekly', 'trip_count_monthly', 'avg_rating_last_90_days',
                    'cancellation_rate', 'age', 'rent_paid_last_12_months',
                    'utility_paid_last_12', 'app_logins_last_30_days', 
                    'complaints_last_6_months', 'outstanding_loans']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    return df

def predict_credit_score(user_data: dict) -> float:
    """
    Take raw user data dict, preprocess, and return predicted credit score.
    """
    df_processed = preprocess_user_input(user_data)
    prediction = pipeline.predict(df_processed)[0]
    return float(prediction)