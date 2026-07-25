"""

"""
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import messages_to_dict,messages_from_dict
import json

# 1: 创建chatMessageHistory（将对话历史存储到内存）
# 内存中的对话历史容易丢失，因此需要将对话历史保存起来
history = ChatMessageHistory()
history.add_user_message("你好，在吗")
history.add_ai_message("在。")

# 2: 将内存中的对话历史记录到文件中(将message序列化成字典)
dict = messages_to_dict(history.messages)
print(dict)

# 3: 将对话历史保存到文件中
with open("history.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(dict, ensure_ascii=False))

# 4：将对话历史从文件中恢复到内存
with open("history.txt", "r", encoding="utf-8") as f:
    messages = json.loads(f.read())
# messages = json.load()
