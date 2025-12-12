# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/11 10:46
# FileName: scrape_subtitles_save_vector_database
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
"""
langchain集成了B站官方接口，可以爬取视频信息，比如字幕，介绍，
到该网站爬取字幕是为了使用langchain保存数据到向量库，并且将数据库持久化保存在电脑磁盘中，
持久化保存的目的是为了方便后期随时调用向量数据库加载这部分数据，而不用再创建向量数据库.
也可以爬取油管网站的视频字幕，安装方法：pip install youtube-transcript-api pytube
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.combine_documents import create_stuff_documents_chain


# 1.Using large models on the OpenRouter platform
from exts import OpenRouter_api_key, TY_API_KEY

# 2.准备好数据链接 Prepare the data link
l_bilibili_vido = ["https://www.bilibili.com/video/BV1chBeYNEQR?t=9.3"]

# 3.手动获取B站视频网页中cookie中的sessdata， bili_jct， buvid3
# Manually obtain sessdata, bili_jct, and buvid3 from the cookies in the Bilibili video webpage.

# 4.爬取视频链接的数据并加载为document对象 Crawl the data of the video link and load it as a document object
document_obj = BiliBiliLoader(
    video_urls=l_bilibili_vido,
    sessdata="c3cd115c%2C1781059133%2Cab659%2Ac2CjDl35nH4gsrMpnkNHQTAWvsr4n_ifUWAo8ZnJ1Em_kVraYfR34CECyKozYhNGyQfK4SVldQOWhIaXc2bFg1TEhjZ2hWNDF4Snk1LUU4NjNXQ3pvV2FLN1h5cGZ1ZVRQYnQwdkpXVGF3ektTVGNTWElCanJRV0lVT3hMRVF5YTlqeFM2NWt2SmZ3IIEC",
    bili_jct="b31075887c9375172450e3711312fe3b",
    buvid3="6C009ED2-69D6-DC59-D098-831AEF04AA6418147infoc"
).load()

# 5.使用文档分割器进行分割 Split using a document splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用RecursiveCharacterTextSplitter切割视频内容为数据块 Use RecursiveCharacterTextSplitter to split video content into data chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

data_chunks = text_splitter.split_documents(document_obj)

# 6.持久化到向量空间 Persist to vector space
from langchain_chroma import Chroma

# 调用大模型API Call the large model API
from langchain_openai import ChatOpenAI

"""
"init_chat_model" is the "unified initialization tool" of LangChain. However, the model_provider list of this tool 
does not include OpenRouter by default. Therefore, we need to use ChatOpenAI (because the interface of OpenRouter 
is compatible with OpenAI) to indirectly call it. 初始化模型（用ChatOpenAI适配OpenRouter）
"""
qw_model = ChatOpenAI(
    model="qwen/qwen3-235b-a22b:free",  # OpenRouter上的模型名
    openai_api_key=OpenRouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1"  # OpenRouter的API地址
)

# 持久化路径 Persistence path
persist_dirpath = r"./vector_persist/bilibili_captions_by_openRouter"

# 判断或创建持久化目录路径 Determine or create a persistent directory path
if not os.path.exists(persist_dirpath):
    os.makedirs(persist_dirpath)

# 判断文档块数据是否有值 Determine if the document block data has a value
if not data_chunks:
    raise "文档块数据为空！"

"""
7.导入向量空间时，对document对象做数据筛选，使用filter_complex_metadata方法筛选掉不支持得数据格式，
不然报错Try filtering complex metadata from the document using langchain_community.vectorstores.utils.filter_complex_metadata.
(When importing the vector space, 
perform data filtering on the document object and use the filter_complex_metadata method to filter out unsupported data formats.)
"""
from langchain_community.vectorstores.utils import filter_complex_metadata

data_chunks_filter = filter_complex_metadata(data_chunks)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=TY_API_KEY
)

# 向量空间 vector space
vector = Chroma.from_documents(documents=data_chunks_filter, embedding=embeddings, persist_directory=persist_dirpath)

# 检索器 Create a retriever
retrieve_obj = vector.as_retriever()

# 系统提示模板 system prompt
s_prompt = """
你是一个做问答的助手，使用一下检索到的上下文片段来回答问题，如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁。\n{context}
"""

# 创建提示模板 Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", s_prompt),
        ("human", "{input}")
    ]
)

# 创建用户提问的chain Create a chain for user questions
user_question_chain = create_stuff_documents_chain(qw_model, prompt)

# 创建带有检索器的chain Create a chain with a retriever
retriever_chain = create_retrieval_chain(retrieve_obj, user_question_chain)

# 用户提问 user`s question
user_ask = {'input': "哪句字幕最搞笑？"}

# 运行提问 Run the question
resp = retriever_chain.invoke(user_ask)

# print(resp.get('answer'))
"""
Result as the follow：
最搞笑的是“两面包哈哈啊 这是牛牛肉”。  
这句话无厘头又重复，显得特别滑稽。  
其他如“抱团有点快啊”也有趣，但不如这句突出。
"""
