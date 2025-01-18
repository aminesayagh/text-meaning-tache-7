import boto3
import os
from botocore.exceptions import ClientError

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.raw_bucket = os.getenv('S3_RAW_BUCKET')
        self.processed_bucket = os.getenv('S3_PROCESSED_BUCKET')
        self.models_bucket = os.getenv('S3_MODELS_BUCKET')

    def upload_file(self, file_path: str, bucket: str, object_name: str = None) -> bool:
        """Upload a file to S3 bucket"""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, bucket, object_name)
            return True
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False

    def download_file(self, bucket: str, object_name: str, file_path: str) -> bool:
        """Download a file from S3 bucket"""
        try:
            self.s3_client.download_file(bucket, object_name, file_path)
            return True
        except ClientError as e:
            print(f"Error downloading file from S3: {e}")
            return False

    def list_files(self, bucket: str, prefix: str = '') -> list:
        """List files in S3 bucket with optional prefix"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            print(f"Error listing files in S3: {e}")
            return []

    def save_text_data(self, text: str, filename: str, is_processed: bool = False) -> bool:
        """Save text data to appropriate bucket"""
        bucket = self.processed_bucket if is_processed else self.raw_bucket
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=filename,
                Body=text.encode('utf-8')
            )
            return True
        except ClientError as e:
            print(f"Error saving text data to S3: {e}")
            return False