"""
langchain中常用的结构化解释方式：
    切记：让llm按照用户的要求的格式输出返回结果（提示词）。
    1：StrOutputParser默认是生成的字符串格式的数据（大语言模型输出数据的格式）
    2：CommaSeparatedListOutputParser逗号分割的列表解析数据
    3：JsonOutputParser json格式的解析器
    4：PydanticOutputParser：PydanticOutputParser结构的数据（校验）
"""
import os
from functools import lru_cache
from typing import Any
from fasttext import FastText
from flask.cli import load_dotenv
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser, JsonOutputParser, \
    PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from passlib.utils import des
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

def demo01():
    """
    StrOutputParser默认是生成的字符串格式的数据（大语言模型输出数据的格式）
    作用：将AIMessage对象转换成纯字符串类型
    :return:
    """
    model = get_chat_model()

    # 定义解析器
    parser = CommaSeparatedListOutputParser()

    # 定义提示词模版
    prompt = ChatPromptTemplate.from_template(
        "请用中文列出来{topic}的五个最重要的特点："
    )

    # langchain中可以将多个组件使用lcel的语法结构连接多个组件
    chain = prompt | model | parser
    # 调用模型
    result = chain.invoke({
        "topic": "大模型"
    })

    print(result)

def demo02():
    """
    CommaSeparatedListOutputParser逗号分割的列表解析数据
    :return:
    """
    model = get_chat_model()

    # 定义解析器
    parser = CommaSeparatedListOutputParser()

    # 获取parser的格式化指令
    format = parser.get_format_instructions()

    # prompt = (f"请用中文列出来大模型的五大最重要的特点：\n"
    #           "你需要按照逗号连接每一个, "
    #             "eg: `foo, bar, baz` or `foo,bar,baz`")
    # result = model.invoke(prompt)

    # 定义提示词模版
    prompt = ChatPromptTemplate.from_template(
        "请用中文列出来{topic}的五个最重要的特点：\n{format_instructions}"
    )

    # langchain中可以将多个组件使用lcel的语法结构连接多个组件
    chain = prompt | model | parser
    # 调用模型
    result = chain.invoke({
        "topic": "大模型",
        "format_instructions": format
    })

    print(result)

def demo03():
    """
    JsonOutputParser解析器，让LLM返回JSON格式的数据
    :return:
    """
    model = get_chat_model(streaming=False)

    # 定义解析器
    parser = JsonOutputParser()

    # 获取格式化指令
    format = parser.get_format_instructions()

    # 定义提示词
    prompt = ChatPromptTemplate.from_template(
        "生成一个包含了{person}的基本信息的json，包含：姓名、职业、年龄和头衔列表，不要包含注释和额外的说明。\n{format_instructions}"
    )

    # langchain中可以将多个组件使用lcel的语法结构连接多个组件
    chain = prompt | model | parser
    # 调用模型
    result = chain.invoke({
        "person": "雷军",
        "format_instructions": format
    })

    print(result)

def demo04():
    """
    PydanticOutputParser：PydanticOutputParser结构的数据（校验）
    :return:
    """
    model = get_chat_model(streaming=False)

    # 定义一个pydantic类
    class Movie(BaseModel):
        title:str = Field(description="电影标题")
        director:str = Field(description="导演姓名")
        year:int = Field(description="上映年份")
        genre:str = Field(description="电影类型")
        rating:float = Field(description="电影评分（1-10分）")

    # 定义解析器
    parser = PydanticOutputParser(pydantic_object=Movie)

    # 获取格式化指令
    format = parser.get_format_instructions()

    # 定义提示词
    prompt = ChatPromptTemplate.from_template(
        "生成一部{genre}电影的信息，\n{format_instructions}"
    )

    # langchain中可以将多个组件使用lcel的语法结构连接多个组件
    chain = prompt | model | parser
    # 调用模型
    result = chain.invoke({
        "genre": "科幻",
        "format_instructions": format
    })

    print(result)

if __name__ == '__main__':
    demo02()





