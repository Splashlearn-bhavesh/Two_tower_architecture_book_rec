import os
import json

import pandas as pd
import pickle
from typing import List, Dict, Optional
from src.constants import BASE_FOLDER

import numpy as np
import os
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('side')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'side.log'), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def read_sql_file(file_name: str) -> str:
    """
    Reads the contents of an SQL file.

    Parameters:
    file_name (str): The path to the SQL file.

    Returns:
    str: SQL query as a string.

    Raises:
    FileNotFoundError: If the file does not exist.
    IOError: If an I/O error occurs.
    """
    logger.debug(f"Attempting to read SQL file: {file_name}")
    
    if not os.path.exists(file_name):
        logger.error(f"File not found: {file_name}")
        raise FileNotFoundError(f"SQL file '{file_name}' does not exist.")
    
    try:
        with open(file_name, 'r') as f:
            sql_query = f.read()
            logger.info(f"Successfully read SQL file: {file_name}")
            return sql_query
    except IOError as e:
        logger.exception(f"Error reading file {file_name}: {e}")
        raise

def save_dataframe(df: pd.DataFrame, filename: str, index: bool = False, mode: str = 'w'):
    """
    Saves a DataFrame to a CSV file with logging and error handling.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The path to the CSV file.
    - index (bool): Whether to write row names (index). Default is False.
    - mode (str): File mode, 'w' to overwrite, 'a' to append. Default is 'w'.

    Raises:
    - IOError: If saving the file fails.
    """
    logger.debug(f"Preparing to save DataFrame to '{filename}' (mode={mode}, index={index})")

    if os.path.exists(filename) and mode == 'w':
        logger.warning(f"File '{filename}' will be overwritten.")

    try:
        df.to_csv(filename, index=index, mode=mode)
        logger.info(f"DataFrame successfully saved to '{filename}'")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame to '{filename}': {e}")
        raise IOError(f"Could not save DataFrame to '{filename}'") from e
    



def save_dict_to_json(data_dict, file_name):
    """
    Saves dictionary into JSON file inside BASE_FOLDER.
    If file exists, merges existing content with new data.

    Parameters:
    - data_dict (dict): Dictionary to save.
    - file_name (str): Name of JSON file (without folder path).
    """
    # Full path to file
    file_path = os.path.join(BASE_FOLDER, file_name)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Load existing data if file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, dict):
                raise ValueError("Existing JSON is not a dictionary.")
        except (json.JSONDecodeError, ValueError):
            existing_data = {}
    else:
        existing_data = {}

    # Merge dictionaries (new data overrides old keys)
    
    # Save merged data
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    print(f"Dictionary saved and merged into {file_path}")


def load_json_file(file_path):
    """
    Loads a JSON file from the given file_path.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Parsed JSON content (empty dict if file does not exist or is invalid).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {}




def fast_split_user_interactions(df, user_col="user_id", item_col="item_id", time_col="timestamp", min_interactions=10, seed=42):
    np.random.seed(seed)

    # Sort once
    df = df.sort_values([user_col, time_col])

    # Count interactions per user
    df["interaction_count"] = df.groupby(user_col)[user_col].transform("count")

    # Filter eligible users
    eligible_mask = df["interaction_count"] > min_interactions

    # Rank interactions per user (latest = highest rank)
    df["rank_desc"] = df.groupby(user_col)[time_col].rank(method="first", ascending=False)

    # Pick eligible users for validation or test
    eligible_users = df.loc[eligible_mask, user_col].drop_duplicates()
    val_users = eligible_users.sample(frac=0.5, random_state=seed)
    test_users = eligible_users[~eligible_users.isin(val_users)]

    # Assign split
    df["split"] = "train"
    df.loc[(df[user_col].isin(val_users)) & (df["rank_desc"] <= 4), "split"] = "val"
    df.loc[(df[user_col].isin(test_users)) & (df["rank_desc"] <= 4), "split"] = "test"

    # Drop helper cols
    df = df.drop(columns=["interaction_count", "rank_desc"])

    # Return split DataFrames
    return (
        df[df["split"] == "train"].drop(columns="split"),
        df[df["split"] == "val"].drop(columns="split"),
        df[df["split"] == "test"].drop(columns="split")
    )
