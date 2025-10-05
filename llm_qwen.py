#!/usr/bin/env python3
"""
Qwen LLM 客户端
基于阿里云 DashScope 文本生成接口

使用环境变量配置：
- QWEN_API_KEY: DashScope API Key
- QWEN_MODEL: 模型名称，默认 qwen-max
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Api-Key": api_key,
    }


def generate_reply(user_text: str, system_prompt: Optional[str] = '返回50个字以内的简短回答',
                   model: Optional[str] = None, timeout: float = 15.0) -> Optional[str]:
    """
    调用 Qwen 接口，根据识别文本生成回复

    Args:
        user_text: 用户输入文本
        system_prompt: 系统提示（可选）
        model: 模型名称（可选，默认环境变量 QWEN_MODEL 或 qwen-max）
        timeout: 请求超时时间

    Returns:
        str or None: 生成的文本回复
    """
    api_key = os.getenv("QWEN_API_KEY", "sk-42c69392f5a14dfdb1df927530d892d7")
    if not api_key:
        logger.error("QWEN_API_KEY 未设置，无法调用大模型")
        return None

    model_name = model or os.getenv("QWEN_MODEL", "qwen-max")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"text": system_prompt}]})
    messages.append({"role": "user", "content": [{"text": user_text}]})

    payload: Dict[str, Any] = {
        "model": model_name,
        "input": {"messages": messages},
        "parameters": {"result_format": "message"},
    }

    try:
        resp = requests.post(
            DASHSCOPE_URL,
            headers=_build_headers(api_key),
            data=json.dumps(payload),
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.error("Qwen 请求失败: %s - %s", resp.status_code, resp.text)
            return None

        data = resp.json()
        content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
        # content 可能是列表或字符串
        if isinstance(content, list) and content:
            # 取第一个 text 字段
            for part in content:
                text = part.get("text")
                if text:
                    return text
        if isinstance(content, str):
            return content
        return None
    except Exception as e:
        logger.error("调用 Qwen 失败: %s", e)
        return None


