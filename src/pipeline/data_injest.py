import pandas as pd
from sqlalchemy import create_engine

def ingest_data(data_path, username, password, database):
    host = "localhost"
    port = 5432
    # Read data from CSV
    data = pd.read_csv(data_path)
    # Prepare PostgreSQL connection string
    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    # Create SQLAlchemy engine
    engine = create_engine(connection_str)
    # Insert data into PostgreSQL (replace table if it exists)
    data.to_sql('Nova_Partner_Data', engine, if_exists='replace', index=False)
    print("Data ingested into PostgreSQL table: Nova_Partner_Data")

