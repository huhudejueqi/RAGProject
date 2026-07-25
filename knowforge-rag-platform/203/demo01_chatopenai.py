"""
langchain中使用model的几种方式
1：chatopenai：调用openai接口访问大语言模型（千问、deepseek）
2：ollamaLLM：调用本地部署的大语言模型
3：chatollama+message（聊天对话）：带有systemprompt的多轮对话
4：ollamaEmbedding：文本转向量模型

使用步骤：
1：创建anaconda虚拟环境（安装了python3.12和requirements.txt）
2：在pycharm中使用创建的虚拟环境（程序运行在了这个虚拟环境中）
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI

# 加载配置文件信息
load_dotenv()

"""
读取环境变量的时候经常会出现的问题
问题：如果在windows中配置了环境变量，名字也叫做DASHSCOPE_API_KEY，那么会优先使用环境变量配置的
解决：
    1：删除环境变量配置的DASHSCOPE_API_KEY
    2：取一个别的名字
"""
print(os.environ.get("DASHSCOPE_API_KEY"))
def demo01():
    """
    ChatOpenAI兼容了接口调用商用大语言模型，可以使用千问、deepseek等等主流的大语言模型
    model：模型的名称
    base_url：api调用的地址
    openai_api_key：api调用的密钥
    max_tokens：限制单词生成的最大token数量
    temperature：控制随机性（0：确定性最高，1：最随机）
    :return:
    """
    model = ChatOpenAI(
        model = os.environ.get("LLM_MODEL"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        openai_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        max_tokens=1000,
        temperature=0.1
    )

    # 调用大语言模型生成答案（一次输入一次输出）
    result = model.invoke("请给我讲一个笑话")
    print(result.content)

def demo02():
    """
    调用本地部署的ollama的大语言模型
    :return:
    """
    model = OllamaLLM(
        model="qwen3:1.7b"
    )

    result = model.invoke("请给我讲一个笑话")
    print(result)

def demo03():
    """
    调用带有聊天功能的（多个角色）本地部署的ollama的大语言模型
    :return:
    """
    model = ChatOllama(
        model="qwen3:1.7b"
    )

    # 构建多个角色的对话信息
    messages = [
        SystemMessage(content="你是一个著名的诗人，擅长写唐诗宋词"),
        HumanMessage(content="帮我写一首花前月下的唐诗。")
    ]
    result = model.invoke(messages)
    print(result.content)

def demo04():
    """
    将文本转成向量
    :return:
    """
    model = OllamaEmbeddings(
        model="bge-m3"
    )

    # 将一个文本字符串转成向量
    result = model.embed_query("你是一个测试文本")
    print(result)

    # 将多个文本字符串转成向量
    result = model.embed_documents(["你是一个测试文本", "你是另一个测试文本"])
    print(result)

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

def demo05():
    model = get_chat_model(streaming=False)
    # 调用大语言模型生成答案（一次输入一次输出）
    result = model.invoke("请给我讲一个笑话")
    print(result.content)

if __name__ == '__main__':
    demo04()
