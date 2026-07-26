"""知识图谱抽取提示词（中文版）。

参考 graphrag-Chinese-llm 的 prompts_chinese 优化版，
适配本项目使用的 LLM 调用方式。
"""

# ── 实体与关系抽取主提示 ──
ENTITY_RELATION_EXTRACT_PROMPT = """### 角色

你是一个信息提取助手，任务是从文本中识别指定类型的实体及其关系，并以结构化格式输出。

### 目标

根据提供的文本和实体类型列表，识别文本中所有符合指定类型的实体及其明确的关系，返回包含实体和关系的列表。

### 操作步骤

1. **实体识别**：
   - 从文本中识别所有符合指定类型（[{entity_types}]）的实体。
   - 对每个实体提取以下信息：
     - **entity_name**：实体名称，首字母大写。
     - **entity_type**：实体类型，必须是 [{entity_types}] 中的一种。
     - **entity_description**：实体属性和活动的详细描述。
   - 格式化每个实体为：`("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)`。

2. **关系识别**：
   - 从步骤 1 识别的实体中，找出所有**明确相关**的实体对（source_entity, target_entity）。
   - 对每对相关实体提取以下信息：
     - **source_entity**：源实体名称，与步骤 1 一致。
     - **target_entity**：目标实体名称，与步骤 1 一致。
     - **relationship_description**：解释为何认为两实体相关。
     - **relationship_strength**：关系的强度评分（数值 1-10）。
   - 格式化每个关系为：`("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)`。

3. **输出格式**：
   - 以中文返回单一列表，包含步骤 1 和 2 识别的所有实体和关系。
   - 使用 **{record_delimiter}** 作为列表分隔符。
   - 在列表末尾添加 **{completion_delimiter}**。

### 注意事项

- 仅识别符合指定类型（[{entity_types}]）的实体，忽略其他类型。
- 仅提取文本中明确表示的关系，避免推测。
- 确保所有输出条目格式一致，字段完整。

### 示例

**输入**：
- 实体类型：ORGANIZATION, PERSON
- 文本：Verdantis 的 Central Institution 将于周一和周四召开会议，计划于周四下午 1:30 发布最新政策决定，随后由 Central Institution 主席 Martin Smith 主持新闻发布会。投资者预期 Market Strategy Committee 将基准利率维持在 3.5%-3.75%。

**输出**：
```
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}Verdantis 的中央机构，负责设定利率并于周一和周四召开会议)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Central Institution 的主席，将在新闻发布会上回答问题)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}Central Institution 的委员会，负责利率和货币供应增长的关键决策)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith 是 Central Institution 的主席，将在新闻发布会上答问{tuple_delimiter}9)
{completion_delimiter}
```

### 输入数据

```
实体类型：{entity_types}
文本：{input_text}
```

### 操作指令

根据上述指导和输入数据，提取所有符合指定类型的实体和关系，并以指定格式返回。确保不遗漏任何明确的相关实体或关系，不得编造信息。"""

# ── 继续抽取提示（用于多轮迭代抽取）──
CONTINUE_EXTRACT_PROMPT = """### 角色

你是一个信息提取助手，继续从之前的文本中识别遗漏的实体和关系。

### 目标

回顾之前的实体和关系列表，判断是否还有未识别的实体或关系。
如果已完整提取所有实体和关系，请输出 {completion_delimiter}
否则，继续以指定格式输出新发现的实体和关系。

### 输入数据

```
实体类型：{entity_types}
文本：{input_text}
已识别的实体和关系：{existing_output}
```

### 操作指令

分析文本和已识别的信息，找出遗漏的实体和关系。如果已完整提取，只输出 {completion_delimiter}。"""

# ── 描述摘要提示 ──
SUMMARIZE_DESCRIPTIONS_PROMPT = """### 角色

你是一个信息整合助手，负责根据提供的实体和描述数据生成全面的摘要。

### 目标

根据输入的一个或两个实体及其相关的描述列表，将所有描述整合为一个连贯、全面的描述，确保包含所有描述中的信息，并以第三人称视角撰写。

### 操作步骤

1. **分析输入**：
   - 接收实体（{entity_name}）及其相关的描述列表。
   - 确保所有描述均与指定实体相关。

2. **整合描述**：
   - 将描述列表中的所有信息合并为单一描述。
   - 保留所有描述中的关键信息，避免遗漏。
   - 确保描述中提及实体名称，以提供完整上下文。

3. **处理矛盾**：
   - 如果描述间存在矛盾，分析并解决矛盾，生成一个逻辑一致的摘要。
   - 优先考虑最可靠或最详细的描述，并在摘要中说明如何解决矛盾。

4. **输出格式**：
   - 以中文输出，采用第三人称视角。
   - 确保摘要清晰、连贯，语言自然流畅。

### 注意事项

- 不得添加未在描述列表中的信息。
- 确保摘要涵盖所有描述的要点，避免偏向单一描述。
- 如果描述不足以生成全面摘要，明确说明信息局限性。

### 输入数据

```
实体：{entity_name}
描述列表：{description_list}
```

### 操作指令

根据上述指导和输入数据，将描述列表整合为一个全面、连贯的中文摘要，确保包含所有信息并解决任何矛盾。"""

# ── 默认实体类型 ──
DEFAULT_ENTITY_TYPES = ["组织", "人物", "地点", "事件", "概念", "项目", "产品", "文档"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
