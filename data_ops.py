import pandas as pd
import os
import shutil
from data_processing.sampler import Sampler
from data_processing.cloud_handler import load_all_csv_data_from_s3
import configparser 
import pandas as pd
from stratified_ops import split_dataframe
import logging

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

config_path = './config.ini'
config = configparser.ConfigParser()
config.read(config_path)

CONFIGS = {
    'ROOT_PATH': config['data']['ROOT_PATH'],
    'ALL_DATA': config['data']['ALL_DATA'],
    'ALL_DATA_STRATIFIED_PATH': config['data']['ALL_DATA_STRATIFIED_PATH'],
    'SAMPLE_SIZE': config['configuration_params']['SAMPLE_SIZE'],
    'FILTER_DATE': config['configuration_params']['FILTER_DATE'],}
    
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

def prepare_data(CONFIGS, CLOUD_CONFIG=config_path):
    """Prepare data for processing.
    
    Args:
        CONFIGS: AppConfig instance with data paths
        CLOUD_CONFIG: Path to cloud configuration file
    """
    # Create directory for all data
    remove_directory_files(CONFIGS.root_path)
    # Create directory for all data
    create_directory_if_not_exists(CONFIGS.root_path)
    # Create directory for stratified data
    create_directory_if_not_exists(CONFIGS.all_data_stratified_path)
    # Load data from S3
    logger.info("Loading all CSV data from S3 bucket: rolling-data, prefix: data")
    # Load and save new data
    latest_date_processed = CONFIGS.filter_date
    load_and_save_new_data(latest_date_processed, CONFIGS.all_data)
    # Stratify data
    stratified_data = stratify_data(
        CONFIGS.all_data,
        CLOUD_CONFIG,
        CONFIGS.filter_date,
        'posted_date_time',
        None,
        int(CONFIGS.sample_size),
        'sampling'
    )
    split_dataframe(
        stratified_data, 
        fraction=0.1, 
        stratify_column='posted_date_time', 
        save_directory=CONFIGS.all_data_stratified_path, 
        seed=42, 
        file_format='csv'
    )
    return "Data preparation completed."