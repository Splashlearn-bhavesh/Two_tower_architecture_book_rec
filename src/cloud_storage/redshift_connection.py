#Create connection on with retrieving and storing the data
from dotenv import load_dotenv
# from src.logger import logging as logger
import psycopg2
import pandas as pd
import os
from src.utils.main_utils import read_sql_file


import os
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('read_shift')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'read_shift.log'), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

class redshift_connection():
    """
    A class for interacting with AWS S3 storage, providing methods for file management, 
    data uploads, and data retrieval in S3 buckets.
    """
   

    def __init__(self):
        """
        Initializes the SimpleStorageService instance with S3 resource and client
        from the S3Client class.
        """
        self.dbname=os.getenv('REDSHIFT_DB')
        self.user=os.getenv('REDSHIFT_USER')
        self.password=os.getenv('REDSHIFT_PASSWORD')
        self.host=os.getenv('REDSHIFT_HOST')
        self.port=os.getenv('REDSHIFT_PORT', 5439)

    def get_redshift_connection(self):
        """
    Establishes and returns a connection to Redshift.
    """
        try:
            conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port= self.port
            )
            logger.info("Connected to Redshift successfully.")
            return conn
        except Exception as e:
            logger.error("Failed to connect to Redshift: %s", e)
            raise

    def redshift_query_fetching_as_df(self, file_name: str) -> pd.DataFrame:
        """
        Executes the  SQL query and returns a DataFrame.
        """

        query = read_sql_file(file_name)

        conn = None
        try:
            conn = self.get_redshift_connection()
            df = pd.read_sql_query(query, conn)
            logger.info("Query executed successfully, retrieved %d rows.", len(df))
            return df
        except Exception as e:
            logger.error("Error executing query: %s", e)
            raise
        finally:
            if conn:
                conn.close()
                logger.info("Connection closed.")
    
    
    
    