"""
demo06_prompt_templates.py — Prompt 模板的 5 种用法

  demo01：PromptTemplate           — 最简单：单变量文本模板
  demo02：FewShotPromptTemplate    — 带示例：给模型提供输入→输出范例
  demo03：ChatPromptTemplate       — 对话模板：简化的多角色消息构建
  demo04：System + Human 组合      — 双角色：系统指令 + 用户模板
  demo05：MessagesPlaceholder      — 动态历史：不固定数量的对话历史插入

学习要点：
  1. 从简单到复杂的 Prompt 构建方式
  2. 理解 Few-Shot Prompting（通过示例教模型怎么做）
  3. 掌握 MessagesPlaceholder 处理动态对话历史的用法
"""
from dotenv import load_dotenv

load_dotenv()

def demo01():
    """PromptTemplate — 最简单的模板：单变量 + 纯文本。

    用法：定义模板 → format(变量=值) → 获得完整 prompt 文本。
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import OllamaLLM

    model = OllamaLLM(model="qwen3.5:9b")

    # {lastname} 是占位符，format() 时替换
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
    prompt = PromptTemplate(input_variables=["lastname"], template=template)

    # format() 填充变量 → 生成完整 prompt 文本
    prompt_text = prompt.format(lastname="王")
    print(prompt_text)  # "我的邻居姓王，他生了个儿子，给他儿子起个名字"

    result = model.invoke(prompt_text)
    print(result)


def demo02():
    """FewShotPromptTemplate — 带示例的模板（Few-Shot Prompting）。

    通过给模型展示几个"输入→输出"范例，教会它完成同类任务。
    范例在前、实际任务在后——模型会根据范例自动推断格式。

    本例：教模型做反义词任务。
    """
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_ollama import OllamaLLM

    model = OllamaLLM(model="qwen3.5:9b")

    # 4 个范例：告诉模型"输入是什么，输出应该是什么"
    examples = [
        {"word": "开心", "antonym": "难过"},
        {"word": "高", "antonym": "矮"},
        {"word": "帅气", "antonym": "丑陋"},
        {"word": "瘦", "antonym": "胖"},
    ]

    # 范例的格式模板
    example_template = "单词: {word}\n反义词: {antonym}\n"
    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_template,
    )

    # FewShotPromptTemplate 组装：前缀 + 范例 + 后缀
    few_shot_prompt = FewShotPromptTemplate(
        prefix="给出每个单词的反义词",      # 任务说明（在范例之前）
        examples=examples,                   # 范例列表
        example_prompt=example_prompt,       # 范例格式
        suffix="单词: {input} 反义词:",     # 实际任务（在范例之后，{input} 是用户输入）
        input_variables=["input"],
        example_separator="\n",
    )

    prompt_text = few_shot_prompt.format(input="粗")
    print(prompt_text)  # 输出包含 4 个范例 + "单词: 粗 反义词:"
    result = model.invoke(prompt_text)
    print(f'result-->{result}')

def demo03():
    """ChatPromptTemplate.from_template() — 简化的对话模板。

    这是创建单条消息模板的最快方式：一行 from_template() + format_messages()。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import OllamaLLM

    model = OllamaLLM(model="qwen3.5:9b")

    template_str = "帮我讲个关于{name}笑话吧"
    # from_template() 创建 ChatPromptTemplate，自动推断输入变量
    prompt_template = ChatPromptTemplate.from_template(template_str)
    # format_messages() 返回 Message 列表（而非纯文本）
    prompt = prompt_template.format_messages(name="气球")
    print(f'prompt-->{prompt}')

    result = model.invoke(prompt)
    print(f'result-->{result}')


def demo04():
    """SystemMessage + HumanMessagePromptTemplate — 多角色模板。

    这是项目中最常用的 Prompt 构建方式：
      1. SystemMessage 定义角色和行为边界
      2. HumanMessagePromptTemplate 定义用户输入的模板
      3. ChatPromptTemplate.from_messages() 组装
    """
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_ollama import OllamaLLM

    model = OllamaLLM(model="qwen3.5:9b")

    # 系统指令：定义角色
    system_prompt = SystemMessage(content="你是取名专家。")
    # 用户消息模板
    human_template = HumanMessagePromptTemplate.from_template(
        "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字。"
    )
    # 组装：[SystemMessage, HumanMessageTemplate]
    chat_template = ChatPromptTemplate.from_messages([system_prompt, human_template])
    # format_messages() → [SystemMessage("你是取名专家"), HumanMessage("我的邻居姓王...")]
    prompt = chat_template.format_messages(lastname="王")
    print(f'prompt-->{prompt}')

    result = model.invoke(prompt)
    print(f'result-->{result}')


def demo05():
    """MessagesPlaceholder — 动态插入不固定数量的历史消息。

    这是多轮对话的关键机制：
      - 第1轮：history=[] → 占位符处为空
      - 第2轮：history=[Human("开心"), AI("难过"), ...] → 展开为 4 条消息
      - 同一个模板，支持任意轮次的对话历史
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_ollama import OllamaLLM

    model = OllamaLLM(model="qwen3.5:9b")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "给出每个单词的反义词"),
            # MessagesPlaceholder：此处的消息数量不固定
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    # 构建对话历史（4 条消息，2 轮示例）
    # 可以用 (role, content) 元组简写：
    #   ("human", "开心") → HumanMessage
    #   ("ai", "难过")    → AIMessage
    history = [
        ("human", "开心"), ("ai", "难过"),
        ("human", "高"), ("ai", "矮"),
    ]
    prompt = prompt_template.format_messages(history=history, question="富有")
    print(f"prompt-->{prompt}")

    result = model.invoke(prompt)
    print(f'result-->{result}')


if __name__ == '__main__':
    # demo01()  # 最简单模板
    # demo02()  # Few-Shot 模板
    # demo03()  # 对话模板
    demo01()  # System+Human 双角色
    # demo05()     # MessagesPlaceholder 动态历史
