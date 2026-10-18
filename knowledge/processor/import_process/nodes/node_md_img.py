"""MD 图片处理节点：上传图片到 MinIO 并替换路径"""

import os
import base64
import time
import re
import logging
import mimetypes
from pathlib import Path
from typing import List, Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_utils import get_minio_client
from knowledge.configs.minio_config import minio_config
from knowledge.configs.lm_config import lm_config
from knowledge.utils.llm_utils import get_llm_client


class NodeMDImg(BaseNode):
    """
    MD 图片处理节点：
    1. 扫描 MD 中的本地图片引用
    2. 上传图片到 MinIO
    3. 可选：用 VL 模型为图片生成中文描述
    4. 替换 MD 中的图片路径为 MinIO URL
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        # 步骤1：获取输入
        md_content = state.get("md_content")
        md_path = state.get("md_path")
        if not md_content or not md_path:
            raise StateFieldError(field_name="md_content", message="md_content 或 md_path 为空")

        # 步骤2：扫描本地图片
        img_dir = Path(md_path).parent / "images"
        images_info = self._step_2_scan_images(md_content, img_dir)
        self.logger.info(f"发现 {len(images_info)} 张图片")

        if not images_info:
            return state

        # 步骤3：上传图片到 MinIO + 可选 VL 描述
        minio_client = get_minio_client()
        replaced_content = md_content

        for idx, info in enumerate(images_info):
            local_path = info["local_path"]
            info["summary"] = self._describe_image(local_path, info.get("alt_text", ""))
            remote_url = (
                info["image_ref"]
                if info["is_remote"]
                else self._step_3_upload_image(minio_client, local_path, state)
            )
            # 即使 MinIO 暂不可用，也保留原路径并写入视觉描述。
            target_url = remote_url or info["image_ref"]
            old_ref = info["original_ref"]
            summary = info.get("summary", "图片")
            new_ref = f"![{summary}]({target_url})"
            replaced_content = replaced_content.replace(old_ref, new_ref)
            self.logger.info(f"图片 {idx + 1}/{len(images_info)}: 描述完成")

            # 速率限制
            if idx > 0 and idx % self.config.requests_per_minute == 0:
                time.sleep(1)

        # 步骤4：保存处理后的 MD
        state["md_content"] = replaced_content
        new_md_path = Path(md_path).parent / f"{Path(md_path).stem}_processed.md"
        with open(new_md_path, "w", encoding="utf-8") as f:
            f.write(replaced_content)
        state["md_path"] = str(new_md_path)

        self.logger.info(f"MD 图片处理完成，新文件: {new_md_path}")
        return state

    def _step_2_scan_images(self, md_content: str, img_dir: Path) -> List[dict]:
        """扫描 MD 中的图片引用"""
        images_info = []

        # 匹配 Markdown 图片语法: ![alt](path)
        md_img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

        for match in md_img_pattern.finditer(md_content):
            alt_text = match.group(1)
            img_ref = match.group(2)

            if img_ref.startswith("http://") or img_ref.startswith("https://"):
                images_info.append({
                    "original_ref": match.group(0),
                    "alt_text": alt_text,
                    "image_ref": img_ref,
                    "local_path": img_ref,
                    "is_remote": True,
                    "summary": alt_text,
                })
                continue

            # 解析本地路径
            if img_ref.startswith("./") or img_ref.startswith("../"):
                local_path = (Path(md_content).parent if isinstance(md_content, Path) else img_dir.parent) / img_ref
            elif img_ref.startswith("images/"):
                local_path = img_dir.parent / img_ref
            else:
                local_path = img_dir / Path(img_ref).name

            local_path = local_path.resolve()
            if local_path.exists():
                images_info.append({
                    "original_ref": match.group(0),
                    "alt_text": alt_text,
                    "image_ref": img_ref,
                    "local_path": str(local_path),
                    "is_remote": False,
                    "summary": alt_text or Path(img_ref).stem,
                })

        return images_info

    def _step_3_upload_image(self, minio_client, local_path: str, state: ImportGraphState) -> str | None:
        """上传图片到 MinIO，返回远程 URL"""
        if not minio_client:
            self.logger.warning("MinIO 客户端不可用，跳过图片上传")
            return None

        try:
            file_name = Path(local_path).name
            file_title = state.get("file_title", "unknown")
            object_name = f"{minio_config.img_dir}/{file_title}/{file_name}"

            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
            )

            endpoint = minio_config.endpoint
            bucket = minio_config.bucket_name
            return f"http://{endpoint}/{bucket}/{object_name.lstrip('/')}"
        except Exception as e:
            self.logger.error(f"图片上传失败 [{local_path}]: {e}")
            return None

    def _describe_image(self, local_path: str, fallback: str) -> str:
        """使用视觉模型为图片生成可检索的中文描述，失败即终止导入。"""
        try:
            if local_path.startswith(("http://", "https://")):
                image_url = local_path
            else:
                mime = mimetypes.guess_type(local_path)[0] or "image/png"
                encoded = base64.b64encode(Path(local_path).read_bytes()).decode("ascii")
                image_url = f"data:{mime};base64,{encoded}"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用一到两句话描述这张电力设施图片，包含设备、结构和可见文字；不要猜测不可见信息。"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }]
            description = get_llm_client(model=lm_config.vl_model).invoke(messages).content.strip()
            return description or fallback or Path(local_path).stem
        except Exception as exc:
            raise RuntimeError(f"视觉描述生成失败: {local_path}: {exc}") from exc
