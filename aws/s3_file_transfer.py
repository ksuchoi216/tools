from pathlib import Path
import boto3
import os

from loguru import logger


def upload_to_s3(
    bucket: str,
    prefix: str,
    local_path: Path | str,
):
    # check local_path file exist
    if isinstance(local_path, str):
        local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # upload the file to s3
    s3_client = boto3.client("s3")

    # If prefix is a directory (ends with /), append the filename
    s3_key = prefix
    if prefix.endswith("/"):
        s3_key = f"{prefix}{local_path.name}"

    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise


def download_from_s3(
    bucket: str,
    prefix: str,
    local_path: Path | str,
):
    if isinstance(local_path, str):
        local_path = Path(local_path)

    # if directory does not exist, create it
    local_dir = Path(local_path).parent
    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    try:
        s3.download_file(Bucket=bucket, Key=prefix, Filename=str(local_path))
        logger.info("Downloaded from s3://%s/%s to %s", bucket, prefix, local_path)
        return local_path
    except Exception as e:
        raise ValueError(
            f"Failed to download from s3://{bucket}/{prefix} to {local_path}: {str(e)}"
        )


def check_file_in_s3(bucket, prefix):
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix)
        return True
    except Exception as e:
        return False
