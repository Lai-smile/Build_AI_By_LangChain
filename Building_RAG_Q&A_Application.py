# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/19 10:50
# FileName: Building_RAG_Q&A_Application
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
"""
构建RAG问答应用，第一部分，加载数据，分割数据；
RAG是一种检索增强生成技术，简单来说就是利用额外的数据源变成一个增强的大模型；
实现思路：
1.加载：首先，我们需要加载数据，通过DocumentLoader完成；
2.分割：Text splitters讲大型文档分割成更小的块。这对于索引数据和将其传递给模型很有用，因为大块数据更难搜索，并且不适合模型的有限上下文窗口；
3.存储：使用VectorStore和Embeddings模型完成；
4.检索：给定用户输入，使用检索器从储存中检索相关分割；
5.生成：ChatModel/LLM使用包括问题和检索到的数据的提示答案。

Building an RAG question-answering application, Part 1: Loading data and splitting data;
RAG is a retrieval-enhanced generation technology. In simple terms,
it uses additional data sources to become an enhanced large model;
Implementation ideas:
1. Loading: First, we need to load the data, which is accomplished through DocumentLoader;
2. Splitting: Text splitters divide large documents into smaller chunks.
This is useful for indexing the data and passing it to the model, as large chunks are more difficult to search
and not suitable for the limited context window of the model;
3. Storage: This is completed using VectorStore and Embeddings model;
4. Retrieval: Given the user input, use the retriever to retrieve relevant chunks from the storage;
5. Generation: ChatModel/LLM uses a prompt answer including the question and the retrieved data.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain

from exts import OpenRouter_api_key, safe_decode

# create a large model object
llm_obj = ChatOpenAI(
    model="tngtech/deepseek-r1t2-chimera:free",
    api_key=OpenRouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)


# load document
def load_document(file):
    """
    加载文档 load document
    :param file: 文档路径 document path
    :return: 文档加载器对象 the loader of document
    """
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
        return loader

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
        return loader

    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
        return loader

    else:
        print('Document format is not supported!')
        return None


# load document data
data = load_document(
    r"D:\Resource\AI测试数据\Preface to Dimensionality Reduction Attack The Three-Body Law of Future Internet Business.docx"
).load()

# Import the data splitting module
from langchain_text_splitters import RecursiveCharacterTextSplitter

# create a splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the document data into multiple chunks
chunks = splitter.split_documents(data)

# To store in a vector database
from langchain_chroma import Chroma

embeddings = DashScopeEmbeddings(model="text-embedding-v4")

# Embed the segmented data into a vector space and store it.
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 检索器 retriever object
retriever = vectorstore.as_retriever()

# 系统提示模板 system prompt
s_prompt = """
You are a Q&A assistant. Use the retrieved context fragments below to answer questions. 
If you don't know the answer, say you don't know. 
Use at most three sentences and keep the answer concise.\n{context}
"""

# 创建提示模板 create a template prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", s_prompt),
        ("human", "{input}")
    ]
)

# Create a chain for user questions
user_question_chain = create_stuff_documents_chain(llm_obj, prompt)

# Create a chain with a retriever
retriever_chain = create_retrieval_chain(retriever, user_question_chain)

# user`s question
user_ask = {'input': "How is dimensionality reduction manifested?"}

# Run the question
resp = retriever_chain.invoke(user_ask)

print(safe_decode(resp.get('answer')))
"""
Result as the follow: 
Dimensionality reduction is manifested through two main models: independent innovation (
focusing on details to find direct, low-cost paths) and confining competitors to lower dimensions (forcing them to 
operate in less advantageous conditions). It involves shifting from high-dimensional strategic thinking ("zoom in") 
to intensely focused execution ("zoom out") to disrupt traditional advantages. This approach allows emerging entities 
to thrive by simplifying survival conditions and attacking established players' vulnerabilities.
"""
