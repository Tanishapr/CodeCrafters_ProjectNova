"""
Module: data_preprocess.py
Author: Tanisha Priya, Harshita Jangde, Prachi Tavse
Date: 2025-09-08
Description:
    Preprocesses Nova Partner raw data from PostgreSQL by handling missing values,
    applying business-specific transformations, one-hot encoding categorical features,
    and scaling numeric features. Saves the processed dataset back to PostgreSQL.
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

def preprocess(username: str, password: str, database: str):
    """
    Preprocesses raw Nova Partner data.

    Parameters:
        username (str): PostgreSQL username.
        password (str): PostgreSQL password.
        database (str): PostgreSQL database name.

    Returns:
        None
    """
    # Prepare PostgreSQL connection
    host = "localhost"
    port = 5432
    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # Load raw data from PostgreSQL
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Data"', engine)
    if df.empty:
        raise ValueError("Raw data table Nova_Partner_Data is empty or missing!")

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Business-specific adjustments
    if "gender" in df.columns:
        # Reduce earnings for female partners by 30% (example adjustment)
        df.loc[df["gender"] == "Female", "earnings_weekly"] *= 0.7
    if "zone" in df.columns:
        # Increase cancellation rates for rural zones
        df.loc[df["zone"].isin(["Rural East", "Rural West"]), "cancellation_rate"] *= 1.5

    # One-hot encode categorical features
    categorical_cols = ['zone', 'job_type', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Standardize numeric features
    numeric_cols = [
        'earnings_weekly', 'trip_count_monthly', 'avg_rating_last_90_days',
        'cancellation_rate', 'age', 'rent_paid_last_12_months',
        'utility_paid_last_12', 'app_logins_last_30_days', 
        'complaints_last_6_months', 'outstanding_loans'
    ]
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    # Save processed data back to PostgreSQL
    df_encoded.to_sql('Nova_Partner_Processed', engine, if_exists='replace', index=False)

    print("Processed data saved to PostgreSQL table: Nova_Partner_Processed")