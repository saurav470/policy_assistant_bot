"""
Utility functions and helpers for the healthcare backend application.
"""

import os
import uuid
import logging
import random
import string
from datetime import datetime
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import NoCredentialsError

from app.config import settings

logger = logging.getLogger(__name__)


def generate_unique_id() -> str:
    """
    Generate a unique identifier.

    Returns:
        str: Unique identifier
    """
    return str(uuid.uuid4())


def generate_random_string(length: int = 10) -> str:
    """
    Generate a random string of specified length.

    Args:
        length (int): Length of the random string

    Returns:
        str: Random string
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        str: Current timestamp
    """
    return datetime.utcnow().isoformat() + "Z"


def get_current_formatted_date() -> str:
    """
    Get current date in a formatted string.

    Returns:
        str: Formatted date string
    """
    return datetime.utcnow().strftime("%B %d, %Y")


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory_path (str): Path to the directory

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.

    Args:
        filename (str): Name of the file

    Returns:
        str: File extension (including the dot)
    """
    return os.path.splitext(filename)[1]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.

    Args:
        filename (str): Original filename

    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    # Ensure filename is not empty
    if not filename:
        filename = f"file_{generate_random_string()}"

    return filename


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path (str): Path to the file

    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {str(e)}")
        return 0.0


def validate_file_type(filename: str, allowed_extensions: list) -> bool:
    """
    Validate if file has an allowed extension.

    Args:
        filename (str): Name of the file
        allowed_extensions (list): List of allowed extensions

    Returns:
        bool: True if file type is allowed
    """
    file_ext = get_file_extension(filename).lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]


def create_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Create a unique filename with timestamp and UUID.

    Args:
        original_filename (str): Original filename
        prefix (str): Optional prefix for the filename

    Returns:
        str: Unique filename
    """
    timestamp = datetime.utcnow().timestamp()
    unique_id = generate_unique_id()
    file_ext = get_file_extension(original_filename)

    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}{file_ext}"
    else:
        return f"{timestamp}_{unique_id}{file_ext}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def safe_get_env(key: str, default: str = "") -> str:
    """
    Safely get environment variable with default value.

    Args:
        key (str): Environment variable key
        default (str): Default value if key not found

    Returns:
        str: Environment variable value or default
    """
    return os.getenv(key, default)


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if a string is a valid UUID.

    Args:
        uuid_string (str): String to validate

    Returns:
        bool: True if valid UUID
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add when truncating

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove null bytes and other control characters
    text = text.replace("\x00", "")

    return text.strip()


class S3Uploader:
    """S3 upload utility class."""

    def __init__(self):
        """Initialize S3 client."""
        self.s3_client = None
        if all([settings.s3_access_key, settings.s3_secret_key, settings.s3_region]):
            try:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.s3_access_key,
                    aws_secret_access_key=settings.s3_secret_key,
                    region_name=settings.s3_region,
                )
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {str(e)}")

    def upload_file(
        self, file_name: str, file_path: str, content_type: str
    ) -> Optional[str]:
        """
        Upload file to S3.

        Args:
            file_name (str): Name for the file in S3
            file_path (str): Local path to the file
            content_type (str): MIME type of the file

        Returns:
            Optional[str]: S3 URL if successful, None otherwise
        """
        if not self.s3_client or not settings.s3_bucket_name:
            logger.warning("S3 client not configured")
            return None

        try:
            self.s3_client.upload_file(
                file_path,
                settings.s3_bucket_name,
                file_name,
                ExtraArgs={"ContentType": content_type},
            )

            s3_url = f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{file_name}"
            logger.info(f"Successfully uploaded {file_name} to S3")
            return s3_url

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return None
        except Exception as e:
            logger.error(f"Failed to upload {file_name} to S3: {str(e)}")
            return None


# Global S3 uploader instance
s3_uploader = S3Uploader()
