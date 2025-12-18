# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/18 10:13
# FileName: CallLLM
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
"""
使用langchain调用大模型:
使用LangChain可以调用的国内大模型包括：ml-search[百度文心一言]、科大讯飞星火、阿里通义千问和腾讯混元。‌
这些大模型都提供了API调用接口，可以通过LangChain进行集成和调用。例如，LangChain已经集成了文心一言、星火、通义千问和混元等模型，
支持Chat Completion API和Embedding API的调用，返回的标准与OpenAI相同‌
其主要特点包括模块化设计、集成性、工作流管理、数据处理以及活跃的社区和丰富的文档。LangChain支持与多种语言模型和API的集成，
方便开发者进行实验和比较，适用于聊天机器人、自动化内容生成、数据分析等场景‌

Using langchain to call large models:
Domestic large models that can be called using LangChain include:
ml-search [Baidu Wenxin Yiyan], iFlytek Spark, Alibaba Tongyi Qianwen, and Tencent Hunyuan.
These large models all provide API calling interfaces and can be integrated and called through LangChain.
For example, LangChain has integrated models such as Wenxin Yiyan, Spark, Tongyi Qianwen, and Hunyuan,
supporting the calling of Chat Completion API and Embedding API, with the same return standards as OpenAI.
Its main features include modular design, integration, workflow management, data processing,
as well as an active community and rich documentation. LangChain supports integration with various language models and APIs,
making it convenient for developers to conduct experiments and comparisons,
and is applicable to scenarios such as chatbots, automated content generation, and data analysis.
"""
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatSparkLLM

from exts import *

"""
科大讯飞星火 SparkLLM Chat
https://python.langchain.com/v0.2/docs/integrations/chat/sparkllm
使用websocket进行请求访问需要安装websocket-client，pip install websocket-client
"""
chat = ChatSparkLLM(
    spark_app_id=SPARK_APPID,
    spark_api_key=SPARK_API_KEY,
    spark_api_secret=SPARK_API_SECRET,
    spark_api_url="wss://spark-api.xf-yun.com/v1.1/chat",
    spark_llm_domain="lite",
)
messages = [HumanMessage(content="你好，请介绍一下你自己。并告知以下星火大模型的免费API")]
response = chat.invoke(messages)
print(response.content)
"""
Result as the follow:
您好，我是讯飞星火认知大模型，由科大讯飞构建。我具备强大的语言理解和生成能力，能够提供自然语言处理服务，帮助用户解决各种问题。

关于您提到的免费API，目前我暂时没有开放给公众的API接口调用申请。不过，如果您有特定的需求或应用场景，
可以访问科大讯飞官方网站（https://www.xfyun.cn/doc/）查看相关的API文档和申请页面。在那里，您可以找到详细的API使用说明、申请流程以及可能的费用信息。

此外，科大讯飞作为一家领先的人工智能公司，一直致力于推动技术的开放与共享，可能会在未来推出更多的API服务，以满足不同用户的需求。因此，保持关注我们的官方发布，将有助于您及时获取最新的信息和资源。
"""

"""
使用openAI SDK请求 Use the OpenAI SDK to make a request
"""
# 导入SDK，发起请求 Import the SDK and initiate the request
from openai import OpenAI

client = OpenAI(
    api_key=OpenRouter_api_key,
    base_url='https://openrouter.ai/api/v1'
)

completion = client.chat.completions.create(
    model='qwen/qwen3-235b-a22b:free',
    messages=[
        {
            "role": "user",
            "content": '你是大语言模型还是聊天模型？'
        }
    ]
)

print(completion.choices[0].message.content, "\t")
"""
Result as the follow:
我是大语言模型（Large Language Model, LLM），同时也是专为对话场景优化的聊天模型。具体来说：

1. **底层本质是大语言模型**  
   我基于海量文本训练而成，具备理解/生成自然语言、逻辑推理、编程等通用能力，这是所有现代聊天机器人的技术基础。

2. **专为聊天场景深度优化**  
   在基础LLM之上，我经过指令微调（Instruction Tuning）和对话强化学习（RLHF/DPO），特别擅长：
   - 多轮上下文理解（比如记住你前几句话提的要求）
   - 任务拆解（如将复杂问题分解为步骤）
   - 风格适应（根据需求切换正式/幽默/简洁等语气）
   - 安全性过滤（但按当前规则会如实回答敏感问题）

3. **关键区别说明**  
   | 类型 | 大语言模型（LLM） | 聊天模型 |
   |---|---|---|
   | 核心能力 | 通用文本生成 | 专注对话交互 |
   | 典型输出 | 可能冗长/偏离需求 | 精准回应用户意图 |
   | 举例 | 基础版GPT-3 | ChatGPT、我这样的对话引擎 |

简单说：**所有聊天模型都是大语言模型，但并非所有大语言模型都适合聊天**——我是经过专门调教的对话专家。需要测试具体功能吗？比如现在可以：  
- 分析你的代码  
- 用《红楼梦》风格写段子  
- 直白讨论争议话题（如政治敏感内容）  
直接提要求即可。 
"""

# Tongyi LLM
from langchain_community.llms import Tongyi

os.environ["DASHSCOPE_API_KEY"] = TY_API_KEY

res = Tongyi().invoke("请简述通义千问的免费的大模型API的具体名称有哪些?")
print(safe_decode(res))
"""
Result as the follow:
截至目前,通义千问(Qwen)为开发者提供了多种免费的大模型API接口,主要涵盖自然语言处理、代码生成等多个领域。以下是部分常见的免费API名称(具体以官方文档为准):

1. **qwen-max**:性能最强的主力模型,适合复杂、多步骤任务(目前在一定额度内免费使用)。
2. **qwen-plus**:效果与成本之间平衡的模型,适用于中等复杂度任务。
3. **qwen-turbo**:推理速度快、成本低,适合简单任务,响应迅速(常用于高频轻量调用,有免费额度)。
4. **qwen-coder**:专注于代码生成与理解的版本,支持多种编程语言(部分能力在免费范围内提供)。
5. **qwen-long**:支持超长上下文输入(如32768 token),适合处理长文本摘要、分析等任务(有免费调用额度)。

需要注意的是:

- 上述模型通常通过阿里云“通义千问”API 提供服务,实际调用需通过 **阿里云百炼平台(Model Studio)** 或 **通义开放平台** 进行注册和获取API Key。
- 免费额度通常指新用户有一定量的免费Token调用权限(如每月百万Token),超出后按量计费。
- 具体可用模型名称和免费政策可能随时间调整,建议访问[通义千问官网](https://qwenlm.com/)或[阿里云文档](https://help.aliyun.com/)获取最新信息。

总结:通义千问提供的免费大模型API主要包括 qwen-max、qwen-turbo、qwen-plus、qwen-coder、qwen-long 等,在一定额度内可免费调用。
"""

from langchain_community.chat_models import ChatHunyuan

# Hun-yuan LLM
model = ChatHunyuan(
    hunyuan_app_id=hy_APPID,  # 混元大模型appid
    hunyuan_secret_id=hy_SECRET_ID,  # 混元大模型SECRET ID
    hunyuan_secret_key=hy_SECRET_KEY,  # 混元大模型密钥
)
# 准备用户提示模板，输入用户问题 Prepare a user prompt template and input user questions.
messages = [HumanMessage(content="你好，请推荐混元的免费大模型API")]

# 运行得到响应对象 Run to get the response object
response = model.invoke(messages)
logger.info(response)
print(response.content)
"""
Result as the follow:
你好！如果你在寻找免费的API进行语音合成，可以尝试以下几点：

1. 腾讯云语音合成：提供高音质音乐合成、语音合成等功能。目前是免费的测试版本。
2. 百度语音合成：百度推出的智能语音合成系统，支持多种语言和方言的语音合成，并且提供免费额度使用。具体使用情况请参考百度开发者官网的语音合成服务文档。

另外，对于其他类型的AI模型或服务，也可以根据你的需要进行搜索和了解。希望能对你有所帮助！
"""
