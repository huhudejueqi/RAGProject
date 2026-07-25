from functools import lru_cache
from typing import Any
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

@lru_cache(maxsize=2)  # 缓存流式/非流式两个客户端实例，避免每次重建 TCP 连接
def get_chat_model(streaming: bool = False) -> Any:
    """返回带缓存的 ChatOpenAI 客户端。

    `streaming` 是缓存 key 的一部分，所以流式和非流式客户端会分开缓存。
    `maxsize=2` 正好覆盖这两种模式，避免重复创建连接，也避免缓存无限增长。

    参数：
        streaming: 是否返回支持流式 token 输出的客户端。

    返回：
        已按全局配置初始化好的 ChatOpenAI 实例。

    调用顺序：LLM 调用阶段 -> get_chat_model()。
    """
    # 构造 ChatOpenAI 客户端
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        openai_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        temperature=os.environ.get("LLM_TEMPERATURE"),
        timeout=os.environ.get("LLM_TIMEOUT"),
        streaming=streaming,
    )