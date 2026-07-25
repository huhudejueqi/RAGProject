"""
在实际的业务中需要将对话历史保存到数据库中进行存储，后续需要加载数据库中存储的历史聊天记录作为提示词的上下文一起提交到大语言模型
面试题：
    多轮对话中的历史对话信息是如何管理的？
    将所有的对话历史分成最近聊天记录+摘要（LLM生成）结合当前问题一起向大语言模型提问生成的上下文
"""

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 13307           # ← 改端口
MYSQL_USER = "root"
MYSQL_PASSWORD = "root123"    # ← 改密码
MYSQL_DATABASE = "subjects_kg"

MYSQL_CONNECTION_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8"
)

def history_from_session(session_id):
    """
    为指定的会话创建SqlChatMessageHistory适配器
    功能清单：
    1：自动的创建数据库表
    2：add_message：自动序列化message->dict
    3:.message:自动select
    :param session_id:
    :return:
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=MYSQL_CONNECTION_URL,
        table_name="chat_message_history",
    )

def demo_sql_history():
    # 演示两个独立对话的历史消息隔离
    session_a = history_from_session(session_id="session_a")
    session_b = history_from_session(session_id="session_b")

    # 会话A：模拟一轮对话
    session_a.add_message(HumanMessage(content="入职需要哪些流程？"))
    session_a.add_message(AIMessage(content="新人入职第一天通常需要完成身份核验，办公设备领取，制度学习和直属负责人确认入职安排。"))
    session_a.add_message(HumanMessage(content="那审批需要多少时间呢？"))

    # 会话B：模拟一轮对话
    session_b.add_message(HumanMessage(content="合同审批的流程？"))
    session_b.add_message(AIMessage(content="大概需要一周左右的时间。"))

    # 会话A的对话历史
    for message in session_a.messages:
        print(message.content)

    print("-----------------------------")
    # 会话B的对话历史
    for message in session_b.messages:
        print(message.content)

if __name__ == '__main__':
    demo_sql_history()