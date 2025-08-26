import pandas as pd
import logging
import sqlalchemy  # or psycopg2 / pyodbc depending on DB
from typing import Optional

class DBLoader:
    """Load raw data from a database connection."""

    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query

    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            logging.info("Connecting to database...")
            engine = sqlalchemy.create_engine(self.connection_string)
            df = pd.read_sql(self.query, engine)
            logging.info(f"Loaded {df.shape[0]} rows from database")
            return df
        except Exception as e:
            logging.error(f"Error loading data from DB: {e}", exc_info=True)
            return None
