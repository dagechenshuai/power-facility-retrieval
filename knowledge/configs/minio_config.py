"""MinIO 对象存储配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class MinIOConfig:
    """MinIO 对象存储配置"""
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket_name: str = ""
    img_dir: str = ""

    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = os.getenv("MINIO_ENDPOINT", "")
        if not self.access_key:
            self.access_key = os.getenv("MINIO_ACCESS_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("MINIO_SECRET_KEY", "")
        if not self.bucket_name:
            self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
        if not self.img_dir:
            self.img_dir = os.getenv("MINIO_IMG_DIR", "/upload-images")


minio_config = MinIOConfig()
