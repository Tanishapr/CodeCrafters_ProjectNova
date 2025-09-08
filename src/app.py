"""
File: app.py
Author: Tanisha Priya, Harshita Jangde, Prachi Tavse
Date: 2025-09-08
Description:
    Streamlit application for the Nova Partner Credit Model.
    Features:
        - Database ingestion, preprocessing, clustering, and model training
        - SHAP summary plots for trained model
        - Interactive Nova Score prediction with per-input SHAP explanations
"""

import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap

from pipeline.data_injest import ingest_data
from pipeline.data_preprocess import preprocess
from pipeline.data_cluster import cluster
from models_credit.xgboost_credit import train_credit_model
from pipeline.model_prediction import predict_credit_score
from pipeline.shap_credit import plot_shap_summary

# STREAMLIT TABS
tab1, tab2 = st.tabs(["Database & Training", "Predict Nova Score"])

#TAB 1: Database & Training
with tab1:
    st.header("PostgreSQL Database Credentials")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    db_name = st.text_input("Database Name")

    if st.button("Ingest, Preprocess, Cluster & Train"):
        if username and password and db_name:
            csv_path = "../Data/Nova_Partner_Data.csv"
            try:
                # Data ingestion
                st.info("Ingesting data into PostgreSQL...")
                ingest_data(csv_path, username, password, db_name)
                st.success("Data ingested.")

                # Preprocessing
                st.info("Preprocessing data...")
                preprocess(username, password, db_name)
                st.success("Data preprocessed.")

                # Clustering
                st.info("Clustering data...")
                cluster(username, password, db_name)
                st.success("Data clustered.")

                # Model training
                st.info("Training credit model...")
                df_results, metrics, fi, _, _ = train_credit_model(
                    username, password, db_name, return_details=True
                )
                st.success("Model trained and results generated!")

                # Display model performance
                rmse, r2 = metrics
                st.subheader("Model Performance")
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("R²", f"{r2:.2f}")

                # Feature importances
                st.subheader("Top Feature Importances")
                st.bar_chart(fi.set_index("feature").head(10))

                # Sample predictions
                st.subheader("Sample Predictions")
                st.dataframe(df_results.head(20))

                # SHAP summary plots
                st.subheader("SHAP Summary Plot")
                pipeline_path = "models/credit_model.pkl"
                plot_shap_summary(pipeline_path, df_results)

            except Exception as e:
                st.error(f"Pipeline error: {e}")
        else:
            st.warning("Please enter all database credentials.")

# TAB 2: Predict Nova Score
with tab2:
    st.header("Predict Nova Score")

    # Collect inputs
    st.session_state.setdefault('input_dict', {})  # initialize if not exists

    st.session_state['input_dict'] = {
        "earnings_weekly": st.number_input("Weekly Earnings", min_value=0.0, step=100.0),
        "trip_count_monthly": st.number_input("Monthly Trip Count", min_value=0, step=1),
        "avg_rating_last_90_days": st.slider("Avg Rating (90 days)", 0.0, 5.0, 4.5, 0.01),
        "cancellation_rate": st.slider("Cancellation Rate", 0.0, 1.0, 0.02, 0.001),
        "zone": st.selectbox("Zone", ["Urban North", "Urban South", "Rural East", "Rural West"]),
        "job_type": st.selectbox("Job Type", ["Delivery Bike", "Delivery Car", "Delivery Van"]),
        "age": st.number_input("Age", min_value=18, max_value=80, value=30, step=1),
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "rent_paid_last_12_months": st.number_input("Rent Payments (last 12 months)", min_value=0, max_value=12, step=1),
        "utility_paid_last_12": st.number_input("Utility Payments (last 12 months)", min_value=0, max_value=12, step=1),
        "app_logins_last_30_days": st.number_input("App Logins (last 30 days)", min_value=0, step=1),
        "complaints_last_6_months": st.number_input("Complaints (last 6 months)", min_value=0, step=1),
        "outstanding_loans": st.number_input("Outstanding Loans", min_value=0.0, step=100.0)
    }

    # Button to predict
    if st.button("Predict Nova Score"):
        # Access safely from session_state
        input_dict = st.session_state['input_dict']

        # Predict
        score = predict_credit_score(input_dict)
        st.success(f"Predicted Nova Score: {score:.2f}")

        # SHAP explanation
        try:
            pipeline = joblib.load("models/credit_model.pkl")
            X_input = pd.DataFrame([input_dict])
            numeric_cols = X_input.select_dtypes(include=["number"]).columns.tolist()

            xgb_model = pipeline.named_steps["model"]
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_input[numeric_cols])

            st.subheader("SHAP Explanation for Input Features")
            plt.figure(figsize=(10,4))
            shap.force_plot(explainer.expected_value, shap_values[0], X_input[numeric_cols].iloc[0], matplotlib=True)
            st.pyplot(plt.gcf())

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")