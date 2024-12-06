import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
import random
import json
import shutil
from typing import Dict
from data_processing.sampler import Sampler
from data_processing.cloud_handler import load_all_csv_data_from_s3
import configparser 
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
from IPython.display import display, Markdown, Latex

from stratified_ops import split_dataframe

config_path = './config_template.ini'
config = configparser.ConfigParser()
config.read(config_path)

CONFIGS = {
    'ROOT_PATH': config['data']['ROOT_PATH'],
    'ALL_DATA': config['data']['ALL_DATA'],
    'ALL_DATA_STRATIFIED_PATH': config['data']['ALL_DATA_STRATIFIED_PATH'],
    'KNOWLEDGE_BASE_PATH': config['data']['KNOWLEDGE_BASE'],
    'SAMPLE_SIZE': config['configuration_params']['SAMPLE_SIZE'],
    'FILTER_DATE': config['configuration_params']['FILTER_DATE']}
    
def remove_directory_files(directory_path: str) -> None:
    """
    Removes all files in the specified directory.
    
    Args:
    directory_path (str): The path to the directory.
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"All files in directory '{directory_path}' have been removed.")
    else:
        print(f"Directory '{directory_path}' does not exist.")
        
def create_directory_if_not_exists(path: str) -> None:
    """
    Creates a directory if it does not already exist.
    
    Args:
    path (str): The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")


def load_and_save_new_data(latest_date_processed: str, output_path: str) -> None:
    """
    Loads new data from S3, saves it to a CSV file.
    
    Args:
    latest_date_processed (str): The latest date processed.
    output_path (str): The output path for the CSV file.
    """
    new_data = load_all_csv_data_from_s3(latest_date_processed=latest_date_processed)
    new_data.to_csv(output_path, index=False)

def stratify_data(input_path: str, config_path: str, filter_date: str, time_column: str, strata_column: str, initial_sample_size: int, section: str) -> pd.DataFrame:
    """
    Loads data, stratifies it, and returns the stratified sample.
    
    Args:
    input_path (str): The input path to the data CSV file.
    config_path (str): The path to the configuration file.
    filter_date (str): The filter date.
    time_column (str): The time column.
    initial_sample_size (int): The initial sample size.
    section (str): The configuration section.
    
    Returns:
    pd.DataFrame: The stratified data sample.
    """
    all_data = pd.read_csv(input_path, low_memory=False)
    sampler = Sampler(config_path, filter_date=filter_date, time_column=time_column, strata_column=strata_column, initial_sample_size=initial_sample_size, section=section)
    return sampler.stratified_sample(all_data)

def prepare_data(CONFIGS=CONFIGS, CLOUD_CONFIG=config_path):
    # Create directory for all data
    remove_directory_files(CONFIGS['ROOT_PATH'])
    # Create directory for all data
    create_directory_if_not_exists(CONFIGS['ROOT_PATH'])
    # Create directory for stratified data
    create_directory_if_not_exists(CONFIGS['ALL_DATA_STRATIFIED_PATH'])
    # Load and save new data
    latest_date_processed = CONFIGS["FILTER_DATE"]
    load_and_save_new_data(latest_date_processed, CONFIGS['ALL_DATA'])
    # Stratify data
    stratified_data = stratify_data(CONFIGS['ALL_DATA'], CLOUD_CONFIG, CONFIGS["FILTER_DATE"], 'posted_date_time', None, CONFIGS["SAMPLE_SIZE"], 'sampling')
    split_dataframe(
        stratified_data, 
        fraction=0.1, 
        stratify_column='posted_date_time', 
        save_directory=CONFIGS['ALL_DATA_STRATIFIED_PATH'], 
        seed=42, 
        file_format='csv')
    return "Data preparation completed."