"""
使用langchain的结构化输出实现意图识别
"""

import os
from functools import lru_cache
from typing import Any, Literal

from fasttext import FastText
from flask.cli import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser, JsonOutputParser, \
    PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from passlib.utils import des
from pydantic import BaseModel, Field
from dotenv import load_dotenv
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
        timeout=1000,
        streaming=streaming,
    )

# Pydantic结构化输出模型
class IntentLLMDecision(BaseModel):
    """
    Literal枚举类型约束：LLM必须要从这个枚举类型的值中选择合适的值，不允许自由发挥

    相当于告诉LLM：
        你的输出必须要是这样的JSON：{intent:‘GREETING’....}
    """
    intent: Literal[
        "GREETING",         # 问候（“您好”）
        "FAQ_QUERY",        # FAQ类问题（简单的常见问题）
        "KNOWLEDGE_QUERY",  # 知识库查询（复杂业务问题，需要文档检索）
        "FOLLOW_UP",        # 追问（依赖对话历史的问题）
        "HUMAN_SERVICE",    # 需要人工服务
        "OUT_OF_SCOPE",     # 越界问题
        "HR"
    ]
    confidence:float = Field(default=0.6, ge=0.0, le=1.0, description="置信度")
    reason:str  = Field(default="")

def demo_structured_output():
    """
    强制LLM按照Pydantic模型输出JSON，自动校验字段类型和范围
    LLM返回了不在枚举中的intent-》Pydantic自动报错
    LLM返回了>1.0的confidence-》Pydantic自动报错

    适用场景：意图分类、信息提取、命名实体类别
    :return:
    """
    model = get_chat_model(streaming=False)
    # 设置结构化输出
    structured_model = model.with_structured_output(IntentLLMDecision)
    decision = structured_model.invoke(
        [
            SystemMessage(content="你是意图分类助手，只判断用户问题的类型。"),
            HumanMessage(content="用户问题：新人入职流程有哪些步骤?")
        ]
    )

    print("intent:", decision.intent)
    print("confidence:", decision.confidence)
    print("reason:", decision.reason)

if __name__ == '__main__':
    demo_structured_output()
