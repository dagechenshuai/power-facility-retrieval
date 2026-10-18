"""MinIO 对象存储客户端工具"""

import logging
from minio import Minio
from minio.error import S3Error
from knowledge.configs.minio_config import minio_config

logger = logging.getLogger("agent.minio")

_minio_client = None


def get_minio_client() -> Minio | None:
    """获取 MinIO 客户端单例，自动创建 Bucket 并设置公开访问策略"""
    global _minio_client
    if _minio_client is not None:
        return _minio_client

    try:
        _minio_client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=False,
        )

        # 确保 Bucket 存在
        if not _minio_client.bucket_exists(minio_config.bucket_name):
            _minio_client.make_bucket(minio_config.bucket_name)
            logger.info(f"创建 MinIO Bucket: {minio_config.bucket_name}")

        # 设置公开读策略
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
                }
            ],
        }
        _minio_client.set_bucket_policy(minio_config.bucket_name, policy)

        return _minio_client
    except S3Error as e:
        logger.error(f"MinIO 连接失败: {e}")
        return None
