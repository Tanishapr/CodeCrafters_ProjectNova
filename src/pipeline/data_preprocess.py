import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

def preprocess(username, password, database):
    host = "localhost"
    port = 5432
    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # Load raw data from raw table
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Data"', engine)
    if df.empty:
        raise ValueError("Raw data table Nova_Partner_Data is empty or missing!")

    # Fill missing values
    df.fillna(0, inplace=True)

    if "gender" in df.columns:
        df.loc[df["gender"] == "Female", "earnings_weekly"] *= 0.7
    if "zone" in df.columns:
        df.loc[df["zone"].isin(["Rural East", "Rural West"]), "cancellation_rate"] *= 1.5
    
    # One-hot encode categorical columns
    categorical_cols = ['zone', 'job_type', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Scale numeric features
    numeric_cols = ['earnings_weekly', 'trip_count_monthly', 'avg_rating_last_90_days',
                    'cancellation_rate', 'age', 'rent_paid_last_12_months',
                    'utility_paid_last_12', 'app_logins_last_30_days', 
                    'complaints_last_6_months', 'outstanding_loans']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    # Save processed data to processed table
    df_encoded.to_sql('Nova_Partner_Processed', engine, if_exists='replace', index=False)

