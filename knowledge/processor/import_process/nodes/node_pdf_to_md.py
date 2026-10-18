"""PDF → Markdown 转换节点（MinerU API）"""

import os
import time
import zipfile
import shutil
import json
import logging
from pathlib import Path

import requests
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.configs.mineru_config import mineru_config


class NodePDFToMD(BaseNode):
    """PDF 转 Markdown：调用 MinerU API 完成 PDF 结构化提取"""

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # 步骤1：校验输入
        pdf_path = self._step_1_validate(state)

        # 步骤2：上传 PDF 并创建批处理任务
        batch_id = self._step_2_upload_pdf(pdf_path)

        # 步骤3：轮询等待解析完成
        result_zip = self._step_3_poll_result(batch_id)

        # 步骤4：解压并提取 MD 内容
        md_content, md_path = self._step_4_extract_md(result_zip, state)

        state["md_content"] = md_content
        state["md_path"] = md_path
        self.logger.info(f"PDF→MD 转换完成，MD 文件: {md_path}")
        return state

    def _step_1_validate(self, state: ImportGraphState) -> str:
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path")
        if not Path(pdf_path).exists():
            raise FileProcessingError(message=f"PDF 文件不存在: {pdf_path}")
        return pdf_path

    def _step_2_upload_pdf(self, pdf_path: str) -> str:
        """步骤2：获取上传 URL 并上传 PDF"""
        base = mineru_config.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {mineru_config.api_token}"}

        # 获取预签名上传 URL
        self.logger.info("获取 MinerU 上传 URL...")
        file_name = Path(pdf_path).name
        resp = requests.post(
            f"{base}/file-urls/batch",
            json={"files": [{"name": file_name, "is_ocr": False}]},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise FileProcessingError(message=f"MinerU 获取上传 URL 失败: {data.get('msg')}")

        upload_info = data["data"]["files"][0]
        upload_url = upload_info["upload_url"]
        batch_id = upload_info["batch_id"]

        # 上传文件
        self.logger.info(f"上传 PDF 到 MinerU...")
        with open(pdf_path, "rb") as f:
            put_resp = requests.put(upload_url, data=f, timeout=120)
            put_resp.raise_for_status()

        # 触发解析
        self.logger.info(f"触发 MinerU 解析, batch_id={batch_id}")
        resp = requests.post(
            f"{base}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()

        return batch_id

    def _step_3_poll_result(self, batch_id: str) -> Path:
        """步骤3：轮询解析结果，超时 600 秒"""
        base = mineru_config.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {mineru_config.api_token}"}
        poll_url = f"{base}/extract-results/batch/{batch_id}"

        max_wait = 600
        interval = 3
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval

            resp = requests.get(poll_url, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") != 0:
                continue

            state = result.get("data", {}).get("state", "")
            if state == "done":
                self.logger.info(f"MinerU 解析完成 (耗时 {elapsed}s)")
                # 下载结果 ZIP
                download_url = result["data"]["files"][0]["download_url"]
                zip_resp = requests.get(download_url, timeout=120)
                zip_resp.raise_for_status()

                # 保存到临时目录
                temp_dir = Path(self.config.temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                zip_path = temp_dir / f"{batch_id}.zip"
                with open(zip_path, "wb") as f:
                    f.write(zip_resp.content)

                return zip_path

            self.logger.info(f"MinerU 处理中... (已等待 {elapsed}s)")

        raise FileProcessingError(message=f"MinerU 解析超时 ({max_wait}s)")

    def _step_4_extract_md(self, zip_path: Path, state: ImportGraphState) -> tuple:
        """步骤4：解压 ZIP，提取 MD 文件内容"""
        file_dir = Path(state["file_dir"])
        file_dir.mkdir(parents=True, exist_ok=True)
        file_title = state.get("file_title", "document")

        # 解压
        with zipfile.ZipFile(zip_path, "r") as zf:
            root = file_dir.resolve()
            for member in zf.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise FileProcessingError(message=f"非法压缩包路径: {member.filename}")
            zf.extractall(file_dir)

        # 查找或重命名 full.md
        md_name = f"{file_title}.md"
        full_md = file_dir / "full.md"
        target_md = file_dir / md_name

        if full_md.exists():
            full_md.rename(target_md)
        elif target_md.exists():
            pass
        else:
            # 尝试找到任意 .md 文件
            md_files = list(file_dir.glob("*.md"))
            if md_files:
                target_md = md_files[0]

        if not target_md.exists():
            raise FileProcessingError(message="MinerU 结果中未找到 Markdown 文件")

        with open(target_md, "r", encoding="utf-8") as f:
            md_content = f.read()

        return md_content, str(target_md)
