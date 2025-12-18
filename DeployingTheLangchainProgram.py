# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/18 12:23
# FileName: DeployingTheLangchainProgram
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
import uvicorn
from fastapi import FastAPI
from langserve import add_routes

"""
使用LangServe部署应用程序
安装方法：pip install langserve[all]
langserve[all]里面包括了python里面性能最好的框架fastapi

Deploy applications using LangServe
Installation method: pip install langserve[all]
langserve[all] includes fastapi, the highest-performance framework in Python.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from exts import *


# 1.call llm
# Tongyi llm
os.environ["DASHSCOPE_API_KEY"] = TY_API_KEY
ty_model = ChatTongyi()

# 2.Define a prompt template where the template content is of an array type, such as a list.
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Please translate the following content into{language}'),
    ('user', '{text}')
])

# 3.Set a parser for parsing response data
parser = StrOutputParser()

# 4.use chain
use_chain = prompt_template | ty_model | parser

# Create a fastapi application
app = FastAPI(title='my langchain service', version='v1.0', description='translate any sentences by langchain')

# add route
add_routes(
    app,
    use_chain,
    path="/chainDemo"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7482)
