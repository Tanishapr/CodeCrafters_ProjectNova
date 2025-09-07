import pandas as pd
from sqlalchemy import create_engine

def ingest_data(data_path, username, password, database):
    # Read data from CSV
    data = pd.read_csv(data_path)
    print("1")
    # Prepare PostgreSQL connection string
    connection_str = f"postgresql+psycopg2://{username}:{password}@localhost:5432/{database}"
    print("2")
    # Create SQLAlchemy engine
    engine = create_engine(connection_str)
    print(type(engine))
    print(engine.url.get_driver_name())
    # Insert data into PostgreSQL (replace table if it exists)
    data.to_sql('Nova_Partner_Data', engine, if_exists='replace', index=False)
    print("4")
    print("Data ingested into PostgreSQL table: Nova_Partner_Data")

