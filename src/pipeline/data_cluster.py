import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

def cluster(username, password, database):
    host = "localhost"
    port = 5432
    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # Load preprocessed data
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Processed"', engine)
    if df.empty:
        raise ValueError("Processed data table Nova_Partner_Processed is empty or missing!")

    # Drop partner_id if present
    features = df.drop(columns=['partner_id'], errors='ignore')

    # K-Means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    df['cluster'] = clusters

    # Save clustered data back to database
    df.to_sql('Nova_Partner_Clustered', engine, if_exists='replace', index=False)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    # Transform centroids into PCA space for plotting
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis', alpha=0.6)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c="red", marker="X", s=200, label="Centroids")
    
    ax.set_title('K-Means Clustering on Nova Partner Data (PCA-reduced)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    plt.grid(True)

    # Show plot in Streamlit
    st.pyplot(fig)