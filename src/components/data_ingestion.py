from src.cloud_storage.redshift_connection import redshift_connection
from src.constants import (
  ITEM_QUERY,
  USER_QUERY,
  USER_CLASS_QUERY,
  USER_LOCATION_QUERY,
  USER_BOOK_PLATFORM,
  DATA_INGESTION_OUTPUT_USER,
DATA_INGESTION_OUTPUT_ITEM,
)
from src.utils.main_utils import save_dataframe
# from src.logger import logging as logger


import os
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class DataIngestion:
    def __init__(self):
        self.user_query = USER_QUERY
        self.item_query = ITEM_QUERY
        self.user_class_query = USER_CLASS_QUERY
        self.user_location_query = USER_LOCATION_QUERY
        self.user_book_platform = USER_BOOK_PLATFORM
        self.output_user = DATA_INGESTION_OUTPUT_USER
        self.output_item = DATA_INGESTION_OUTPUT_ITEM
        self.connection = redshift_connection()

    def fetching_and_saving_data(self):
        """
        Fetches data from Redshift using predefined SQL queries and saves the
        results to CSV files.

        Returns:
            tuple: DataFrames for user, item, and interaction data.
        """
        logger.info("Starting data ingestion process.")

        try:
            logger.debug("Fetching user data from Redshift.")
            user_df = self.connection.redshift_query_fetching_as_df(self.user_query)
            logger.info("User data fetched successfully.")

            # logger.debug("Fetching user data from Redshift.")
            # user_class_df = self.connection.redshift_query_fetching_as_df(self.user_class_query)
            # logger.info("User data fetched successfully.")

            logger.debug("Fetching user data from Redshift.")
            user_location_df = self.connection.redshift_query_fetching_as_df(self.user_location_query)
            logger.info("User data fetched successfully.")

            logger.debug("Fetching user data from Redshift.")
            user_book_platform_df = self.connection.redshift_query_fetching_as_df(self.user_book_platform)
            logger.info("User data fetched successfully.")
            
            logger.debug("Fetching item data from Redshift.")
            item_df = self.connection.redshift_query_fetching_as_df(self.item_query)
            logger.info("Item data fetched successfully.")
            
        except Exception as e:
            logger.exception("Error occurred while fetching data from Redshift.")
            raise RuntimeError("Data fetching failed.") from e

        try:
            logger.debug(f"Saving user data to {self.output_user}.")
            save_dataframe(user_df, self.output_user)
            
            logger.debug(f"Saving item data to {self.output_item}.")
            save_dataframe(item_df, self.output_item)
            
            logger.debug(f"Saving interaction data to {self.output_interaction}.")
            save_dataframe(interaction_df, self.output_interaction)

            logger.info("All datasets saved successfully.")
        except Exception as e:
            logger.exception("Error occurred while saving data to files.")
            raise RuntimeError("Data saving failed.") from e

        logger.info("Data ingestion process completed.")
        return user_df, item_df
