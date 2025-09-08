"""
Module: shap_credit.py
Author: Tanisha Priya, Harshita Jangde, Prachi Tavse
Date: 2025-09-08
Description:
    Generates SHAP summary and force plots for the Nova Credit Model.
    Can be used to visualize overall feature importance or explain individual predictions.
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

def plot_shap_summary(model_path: str, df: pd.DataFrame):
    """
    Loads a trained pipeline and generates SHAP summary and force plots for numeric features.

    Parameters:
        model_path (str): Path to the trained pipeline (joblib .pkl)
        df (pd.DataFrame): Input dataframe with same features used in training
    """
    # Load the trained pipeline
    pipeline = joblib.load(model_path)

    # Drop columns not used in prediction
    X = df.drop(columns=["credit_score", "cluster", "partner_id"], errors="ignore")
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Extract trained XGBoost model
    xgb_model = pipeline.named_steps["model"]

    # Prepare SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X[numeric_cols])

    # Summary plot (bar)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values.values, X[numeric_cols], plot_type="bar", show=False)
    st.pyplot(plt.gcf())

    # Optional: force plot for first row
    st.subheader("SHAP Force Plot for First Row")
    plt.figure(figsize=(8,3))
    shap.plots.force(shap_values[0], matplotlib=True)
    st.pyplot(plt.gcf())