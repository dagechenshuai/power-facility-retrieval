"""Prompt 模板加载工具"""

import os
from pathlib import Path


def load_prompt(name: str, **kwargs) -> str:
    """
    加载 .prompt 模板文件并替换占位符

    Args:
        name: 模板文件名（不含 .prompt 后缀），如 "answer"
        **kwargs: 模板中的 {key} 占位符键值对

    Returns:
        替换后的 prompt 文本

    Example:
        prompt = load_prompt("answer", context="...", question="什么是风机？")
    """
    # 模板目录相对于此文件的位置: ../../prompts/
    prompts_dir = Path(__file__).absolute().parent.parent.parent / "prompts"
    file_path = prompts_dir / f"{name}.prompt"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()

    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))

    return template
