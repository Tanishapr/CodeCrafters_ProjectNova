"""
Module: data_cluster.py
Author: Your Name
Date: 2025-09-08
Description:
    This module defines a function to perform K-Means clustering on preprocessed 
    partner data. The clustered results are saved back to PostgreSQL, and a 
    PCA-based 2D visualization of clusters is generated and displayed in Streamlit.
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

def cluster(username: str, password: str, database: str):
    """
    Perform K-Means clustering on preprocessed Nova partner data.

    Parameters:
    ----------
    username : str
        PostgreSQL database username.
    password : str
        PostgreSQL database password.
    database : str
        PostgreSQL database name.

    Returns:
    -------
    None
        Saves clustered data to database and displays PCA scatter plot in Streamlit.
    """
    # Create PostgreSQL engine
    host = "localhost"
    port = 5432
    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # Load preprocessed data from database
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Processed"', engine)
    if df.empty:
        raise ValueError("Processed data table Nova_Partner_Processed is empty or missing!")

    # Prepare features (exclude partner_id if present)
    features = df.drop(columns=['partner_id'], errors='ignore')

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    df['cluster'] = clusters

    # Save clustered data back to PostgreSQL
    df.to_sql('Nova_Partner_Clustered', engine, if_exists='replace', index=False)

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    # Transform K-Means centroids into PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis', alpha=0.6)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c="red", marker="X", s=200, label="Centroids")
    ax.set_title('K-Means Clustering on Nova Partner Data (PCA-reduced)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)