import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

def ingest_data(data_path, postgres_username, postgres_password, postgres_host, postgres_port, postgres_database):
    # Read data from CSV
    data = pd.read_csv(data_path)

    # Prepare PostgreSQL connection string
    connection_str = f'postgresql://{postgres_username}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}'

    # Create SQLAlchemy engine
    engine = create_engine(connection_str)

    # Insert data into PostgreSQL (replace table if it exists)
    data.to_sql('Nova_Partner_Data', engine, if_exists='replace', index=False)

    print("Data ingested into PostgreSQL table: Nova_Partner_Data")

if __name__ == "__main__":
    load_dotenv()
    # Load credentials from environment variables
    user = os.getenv('PG_USER')
    password = os.getenv('PG_PASSWORD')
    host = os.getenv('PG_HOST')
    port = os.getenv('PG_PORT')
    database = os.getenv('PG_DATABASE')
    ingest_data('..//..//Data//Nova_Partner_Data.csv', user, password, host, port, database)
