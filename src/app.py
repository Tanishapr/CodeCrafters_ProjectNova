import pandas as pd
import streamlit as st
import pickle
import os
from pipeline.data_preprocess import preprocess
from pipeline.data_cluster import cluster
from pipeline.data_injest import ingest_data  # Import the function
from models_credit.xgboost_credit import train_credit_model
from pipeline.model_prediction import predict_credit_score
tab1, tab2 = st.tabs(["Database Connection", "Predict Nova Score"])

with tab1:
    st.header("PostgreSQL Database Credentials")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    db_name = st.text_input("Database Name")
    
    if st.button("Ingest Data"):
        if username and password and db_name:
            csv_path = "../Data/Nova_Partner_Data.csv"  # The generated CSV path
            try:
                st.info("Ingesting data to PostgreSQL...")
                ingest_data(csv_path, username, password, db_name)
                st.info("data injested.")

                st.info("Preprocessing data...")
                preprocess(username, password, db_name)  # data_preprocess.py 
                st.info("data preprocessed.")

                st.info("Clustering data...")
                cluster(username, password, db_name)  #  data_cluster.py
                st.info("data clustered")

                st.info("Training model...")
                df_results, metrics, fi, bias_before, bias_after = train_credit_model(
                    username, password, db_name, return_details=True
                )
                st.success("Model trained and results generated!")

                st.subheader("Model Performance")
                rmse, r2 = metrics
                st.metric(label="RMSE", value=f"{rmse:.2f}")
                st.metric(label="R²", value=f"{r2:.2f}")

                st.subheader("Top Feature Importances")
                st.bar_chart(fi.set_index("feature").head(10))

                st.subheader("Sample Predictions")
                st.dataframe(df_results.head(20))
            except Exception as e:
                st.error(f"Pipeline error: {e}")
        else:
            st.warning("Enter all database credentials to run pipeline.") 
            
with tab2:
    st.header("Predict Nova Score")
    earnings_weekly = st.number_input("Weekly Earnings", min_value=0.0, step=100.0)
    trip_count_monthly = st.number_input("Monthly Trip Count", min_value=0, step=1)
    avg_rating_last_90_days = st.slider("Avg Rating (90 days)", 0.0, 5.0, 4.5, 0.01)
    cancellation_rate = st.slider("Cancellation Rate", 0.0, 1.0, 0.02, 0.001)
    zone = st.selectbox("Zone", ["Urban North", "Urban South", "Rural East", "Rural West"])
    job_type = st.selectbox("Job Type", ["Delivery Bike", "Delivery Car", "Delivery Van"])
    age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    rent_paid_last_12_months = st.number_input("Rent Payments (last 12 months)", min_value=0, max_value=12, step=1)
    utility_paid_last_12 = st.number_input("Utility Payments (last 12 months)", min_value=0, max_value=12, step=1)
    app_logins_last_30_days = st.number_input("App Logins (last 30 days)", min_value=0, step=1)
    complaints_last_6_months = st.number_input("Complaints (last 6 months)", min_value=0, step=1)
    outstanding_loans = st.number_input("Outstanding Loans", min_value=0.0, step=100.0)

    if st.button("Predict Nova Score"):
        input_dict = {
            "earnings_weekly": earnings_weekly,
            "trip_count_monthly": trip_count_monthly,
            "avg_rating_last_90_days": avg_rating_last_90_days,
            "cancellation_rate": cancellation_rate,
            "zone": zone,
            "job_type": job_type,
            "age": age,
            "gender": gender,
            "rent_paid_last_12_months": rent_paid_last_12_months,
            "utility_paid_last_12": utility_paid_last_12,
            "app_logins_last_30_days": app_logins_last_30_days,
            "complaints_last_6_months": complaints_last_6_months,
            "outstanding_loans": outstanding_loans
        }

        # Call centralized prediction function
        score = predict_credit_score(input_dict)
        st.success(f"Predicted Nova Score: {score:.2f}")