"""
Module: xgboost_credit.py
Author: Your Name
Date: 2025-09-08
Description:
    This module defines a function to train an XGBoost regression model 
    on clustered partner data. The function handles data ingestion from 
    PostgreSQL, preprocessing, model training, evaluation, feature importance 
    extraction, and saving the trained pipeline along with predictions back 
    to the database. Bias mitigation steps are removed for simplicity.
"""

import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_credit_model(username: str, password: str, database: str, return_details: bool = False):
    """
    Train an XGBoost regression model to predict credit scores.

    Parameters:
    username : str
        PostgreSQL database username.
    password : str
        PostgreSQL database password.
    database : str
        PostgreSQL database name.
    return_details : bool, default=False
        If True, returns additional metrics, feature importance, and placeholder bias info.

    Returns:
    df : pd.DataFrame
        Original dataframe with predictions saved to database.
    OR (if return_details=True):
    tuple : (df, (rmse, r2), feature_importances_df, bias_before, bias_after)
        - df: DataFrame with predictions
        - (rmse, r2): Evaluation metrics
        - feature_importances_df: DataFrame with feature importances
        - bias_before, bias_after: placeholders (empty dicts)
    """
    host = "localhost"
    port = 5432
    connection_str = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    df = pd.read_sql('SELECT * FROM "Nova_Partner_Clustered"', engine)
    if df.empty:
        raise ValueError("Clustered data table Nova_Partner_Clustered is empty or missing!")

    # Clean column names and remove duplicates
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    seen = set()
    keep = []
    for c in df.columns:
        key = c.lower() if isinstance(c, str) else c
        if key not in seen:
            seen.add(key)
            keep.append(c)
        else:
            print(f"Warning: dropping duplicate column: {c}")
    df = df[keep]

    # Helper function to find columns by multiple names
    def find_col(df, *candidates):
        lower_map = {c.lower().strip(): c for c in df.columns if isinstance(c, str)}
        for cand in candidates:
            if cand is None:
                continue
            k = cand.lower().strip()
            if k in lower_map:
                return lower_map[k]
        return None

    # Ensure 'gender' column exists
    if find_col(df, "gender") is None:
        gender_ohe_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("gender_")]
        if gender_ohe_cols:
            sub = df[gender_ohe_cols].fillna(0)
            inferred = sub.idxmax(axis=1).str[len("gender_"):].str.replace("_", " ").str.title()
            zero_mask = (sub.sum(axis=1) == 0)
            inferred.loc[zero_mask] = "Other"
            df["gender"] = inferred.astype(str)
            df = df.loc[:, ~df.columns.duplicated()]

    # Generate synthetic 'credit_score'
    np.random.seed(42)
    cluster_col = find_col(df, "cluster")
    if cluster_col is None:
        raise ValueError("Required column 'cluster' is missing!")

    base_score = df[cluster_col].map({0: 400, 1: 650, 2: 850})
    age_col = find_col(df, "age")
    age_effect = df[age_col] * 1.2 if age_col else 0
    income_col = find_col(df, "earnings_weekly", "income", "earnings")
    income_effect = df[income_col] * 0.02 if income_col else 0
    zone_col = find_col(df, "zone")
    zone_effect = 0
    if zone_col:
        zmap = {"urban": 20, "semi-urban": 0, "semi urban": 0, "rural": -20}
        zone_effect = df[zone_col].astype(str).str.lower().map(zmap).fillna(0)

    df["credit_score"] = (base_score + age_effect + income_effect + zone_effect +
                          np.random.normal(0, 15, len(df))).clip(300, 1000)

    # Prepare features (X) and target (y)
    drop_cols = [find_col(df, "credit_score") or "credit_score",
                 cluster_col,
                 find_col(df, "partner_id") or "partner_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["credit_score"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found for training.")

    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numeric_cols)], remainder="drop")

    # Build XGBoost pipeline
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Model evaluation
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"✅ Model Performance: RMSE={rmse:.2f}, R²={r2:.2f}")

    # Feature importance
    try:
        xgb_model = pipeline.named_steps["model"]
        importances = xgb_model.feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    except Exception:
        fi = pd.DataFrame()

    # Save model & predictions
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "credit_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    df.to_sql('Nova_Partner_CreditModel', engine, if_exists='replace', index=False)
    print("Saved predictions to Nova_Partner_CreditModel")

    # Return results
    if return_details:
        bias_before, bias_after = {}, {}  # placeholders
        return df, (rmse, r2), fi, bias_before, bias_after

    return df