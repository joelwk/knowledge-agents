import os
import json
import boto3
import pandas as pd
import logging
from dateutil import tz
from azure.storage.blob import BlobServiceClient
from utils import read_config, setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

config_path='./config.ini'
config_params = read_config(section='cloud_info',config_path=config_path)

logger = logging.getLogger(__name__)
bucket_name = config_params['s3_bucket']
bucket_prefix = config_params['s3_bucket_prefix']

class S3Handler:
    def __init__(self, aws_credentials=config_params, local_data_path='./s3-data', region_name='us-east-1'):
        self.bucket_name = aws_credentials['s3_bucket']
        self.bucket_prefix = aws_credentials['s3_bucket_prefix']
        self.local_path = os.path.normpath(local_data_path) + os.sep
        self.region_name = region_name
        self.s3 = self._create_s3_client(aws_credentials)
        
    def _create_s3_client(self, aws_credentials):
        if aws_credentials.get('profile_name'):
            session = boto3.Session(profile_name=aws_credentials['profile_name'])
        else:
            session = boto3.Session(aws_access_key_id=aws_credentials['aws_access_key_id'],
                                    aws_secret_access_key=aws_credentials['aws_secret_access_key'],
                                    region_name=self.region_name)
        return session.client('s3')
    
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def get_s3_client(self):
        return self.s3

    def file_exists_in_s3(self, s3_key):
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False
            
    def upload_dir(self, dir_key):
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, self.local_path)
                s3_key = os.path.join(dir_key, relative_path).replace(os.sep, '/')
                print(f"Bucket name: {self.bucket_name}, Type: {type(self.bucket_name)}")
                if not local_file.endswith('/') and not local_file.endswith('\\'):
                    if not self.file_exists_in_s3(s3_key):
                        self._upload_file(local_file, s3_key)
                    else:
                         print(f"Skipping {s3_key}, already exists in S3.")

    def download_dir(self, dir_key):
        if not local_file_path:
            local_file_path = os.path.join(self.local_path, os.path.basename(dir_key))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=dir_key):
            for file in result.get('Contents', []):
                self.download_file(file['Key'])

    def download_file(self, s3_key, local_file_path=None, file_type='model', custom_objects=None):
        if not local_file_path:
            local_file_path = os.path.join(self.local_path, os.path.basename(s3_key))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, s3_key, local_file_path)
        print(f"Downloaded file {s3_key} to {local_file_path}")
        if file_type == 'model':
            if custom_objects:
                with custom_object_scope(custom_objects):
                    return load_model(local_file_path)
            else:
                return load_model(local_file_path)
        elif file_type in ['csv', 'parquet']:
            if file_type == 'csv':
                return pd.read_csv(local_file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(local_file_path)

    def _upload_file(self, local_file, key):
        self.s3.upload_file(local_file, self.bucket_name, key)
        logger.info(f"Uploaded file {local_file} to {self.bucket_name}/{key}")
        
def load_all_csv_data_from_s3(bucket=bucket_name, s3_prefix=bucket_prefix, select_board=None, latest_date_processed=None):
    logging.info(f"Loading all CSV data from S3 bucket: {bucket}, prefix: {s3_prefix}")
    connector = S3Handler(config_params)
    s3_client = connector.get_s3_client()
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    except Exception as e:
        logging.error(f"Failed to list objects in S3 bucket: {e}")
        return pd.DataFrame()
    
    if 'Contents' not in response:
        logging.info("No files found in the specified bucket and prefix.")
        return pd.DataFrame()

    if latest_date_processed:
        latest_date_processed = pd.to_datetime(latest_date_processed, utc=True)
        # Convert latest_date_processed to UTC if it's not timezone-aware
        if latest_date_processed.tzinfo is None or latest_date_processed.tzinfo != tz.UTC:
            latest_date_processed = latest_date_processed.astimezone(tz.UTC)
        logging.info(f"Latest date processed: {latest_date_processed}")

    # Filter files based on board prefix and latest_date_processed
    filtered_files = []
    for item in response.get('Contents', []):
        if item['Key'].endswith('.csv'):
            # If select_board is None, include all files
            if select_board is None or f'chanscope_{select_board}' in item['Key']:
                # Filter by latest_date_processed if provided
                if latest_date_processed is None or item['LastModified'].astimezone(tz.UTC) > latest_date_processed:
                    filtered_files.append(item)

    csv_objects = [item['Key'] for item in filtered_files]

    if not csv_objects:
        logging.info("No new CSV files to process since the last update.")
        return pd.DataFrame()
    
    all_data_frames = []
    for file_key in csv_objects:
        try:
            temp_file_path = f"temp_{file_key.replace('/', '_')}"
            s3_client.download_file(bucket, file_key, temp_file_path)
            df = pd.read_csv(temp_file_path, low_memory=False)  # Use low_memory=False to address DtypeWarning
            all_data_frames.append(df)
        except Exception as e:
            logging.error(f"Error processing file {file_key}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    combined_data = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()

    logging.info(f"Combined data contains {len(combined_data)} rows.")
    return combined_data

class BlobHandler:
    def __init__(self, connection_string, container_name, local_data_path='../data-azure'):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.container_client = self._create_container_client(connection_string, container_name)
        self.local_path = os.path.normpath(local_data_path) + os.sep

    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _create_container_client(self, connection_string, container_name):
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        return blob_service_client.get_container_client(container_name)

    def _upload_file(self, local_file, blob_name):
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_file, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            self.logger.info(f"Uploaded file {local_file} to Blob Storage as {blob_name}")
        except Exception as e:
            self.logger.error(f"Failed to upload {local_file} to Blob Storage as {blob_name}. Error: {e}")

    def file_exists_in_blob(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception as e:
            return False

    def upload_dir(self, azure_dir_key):
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, self.local_path)
                blob_name = os.path.join(azure_dir_key, relative_path).replace(os.sep, '/')
                if not self.file_exists_in_blob(blob_name):
                    self._upload_file(local_file, blob_name)
                else:
                    print(f"Skipping {blob_name}, already exists in Blob Storage.")

    def download_blob(self, blob_name, local_file):
        blob_client = self.container_client.get_blob_client(blob_name)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with open(local_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        self.logger.info(f"Downloaded blob {blob_name} to {local_file}")

    def download_json_blob(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            blob_data = blob_client.download_blob().readall()
            json_data = json.loads(blob_data)
            self.logger.info(f"Successfully downloaded and parsed JSON blob: {blob_name}")
            return json_data
        except Exception as e:
            self.logger.error(f"Failed to download or parse JSON blob: {blob_name}. Error: {e}")
            return None