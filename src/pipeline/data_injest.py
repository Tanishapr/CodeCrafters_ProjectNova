"""
Module: data_ingest.py
Author: Tanisha Priya, Harshita Jangde, Prachi Tavse
Date: 2025-09-08
Description:
    Reads a CSV file containing Nova Partner data and ingests it into
    a PostgreSQL database using SQLAlchemy. The target table is
    'Nova_Partner_Data'. Existing table is replaced if present.
"""

import pandas as pd
from sqlalchemy import create_engine

def ingest_data(data_path: str, username: str, password: str, database: str):
    """
    Ingests CSV data into a PostgreSQL database.
    
    Parameters:
        data_path (str): Path to the CSV file.
        username (str): PostgreSQL username.
        password (str): PostgreSQL password.
        database (str): PostgreSQL database name.

    Returns:
        None
    """
    # Read CSV data into a pandas DataFrame
    data = pd.read_csv(data_path)

    # Prepare PostgreSQL connection string
    connection_str = f"postgresql+psycopg2://{username}:{password}@localhost:5432/{database}"

    # Create SQLAlchemy engine
    engine = create_engine(connection_str)

    # Insert data into PostgreSQL, replacing table if it exists
    data.to_sql('Nova_Partner_Data', engine, if_exists='replace', index=False)

    print("Data ingested into PostgreSQL table: Nova_Partner_Data")

