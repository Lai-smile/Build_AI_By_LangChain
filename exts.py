# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2024/12/3 14:59
# FileName: exts
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
"""
插件脚本，创建各类大模型对象
"""
import os
import sys
import logging
import importlib

from langchain_community.chat_models import ChatTongyi

key_path = r"D:\Resource\keys\api_keys.py"
sys.path.append(os.path.dirname(key_path))
api_keys_name = os.path.basename(key_path).split('.')[0]
api_keys = importlib.import_module(api_keys_name)

TY_API_KEY = api_keys.ty_api_key

# 通义千问
os.environ["DASHSCOPE_API_KEY"] = TY_API_KEY
ty_model = ChatTongyi()

# openrouter
OpenRouter_api_key = api_keys.openrouter_api_key

def log_func(root_path):
    """
    项目运行日志
    :param root_path:项目根路径
    :return:
    """
    log_name = 'py_log'  # 日志名称(默认py_log)
    log_level = logging.DEBUG  # 日志等级(默认DEBUG)
    console = True  # 是否在控制台输出(默认在控制台输出)
    # 日志输出格式
    log_format = u'[%(asctime)s] - [%(filename)s :%(lineno)d line] - %(levelname)s: %(message)s'
    # 日志文件的名称和路径
    log_file_path = os.path.join(root_path, "log\\" + 'py_log_file.txt')
    # 日志文件夹
    log_dir_path = os.path.dirname(log_file_path)

    if not os.path.exists(log_dir_path):  # 判断文件夹是否存在
        os.makedirs(log_dir_path)  # 创建日志文件夹

    py_loges = logging.getLogger(log_name)  # 创建log
    py_loges.setLevel(log_level)  # 设置等级

    # 定义log格式
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(log_file_path)  # 创建文件
    file_handler.setFormatter(formatter)

    # 遍历多个日志级别
    for level_content in [logging.DEBUG, logging.ERROR, logging.INFO]:
        file_handler.setLevel(level_content)  # 设置写入日志文件的内容

    py_loges.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()  # 创建控制台
        console_handler.setFormatter(formatter)
        py_loges.addHandler(console_handler)

    with open(log_file_path, "a+") as f:
        f.truncate()

    return py_loges


# 创建日志对象
logger = log_func(r"D:\Project\AIProject\LangchainStudy")

def safe_decode(text, fallback_encoding="utf-8"):
    """
    安全解码文本，处理各种编码异常 Safely decode text and handle various encoding exceptions
    :param text: 输入文本（可能是str或bytes） Input text (maybe str or bytes)
    :param fallback_encoding: 兜底编码格式
    :return: 处理后的str文本 The processed string text
    """
    # 如果是bytes类型，先尝试解码 If it is of the bytes type, first try to decode it.
    if isinstance(text, bytes):
        for encoding in [fallback_encoding, "gbk", "gb2312", "latin-1"]:
            try:
                return text.decode(encoding)
            except UnicodeDecodeError:
                continue
        # 所有编码都失败时，忽略错误字符 When all encodings fail, ignore the error characters.
        return text.decode(fallback_encoding, errors="replace")

    # 如果是str类型，处理无法编码的字符（如特殊符号） If it is of type str, handle unencodable characters (such as special symbols)
    elif isinstance(text, str):
        # 去除无法编码的控制字符，或替换为安全字符 Remove unencodable control characters or replace them with safe characters
        cleaned_text = unicodedata.normalize("NFKC", text)
        # 尝试用当前终端编码输出，失败则用utf-8 Try to output using the current terminal encoding; if it fails, use utf-8.
        terminal_encoding = sys.stdout.encoding or fallback_encoding
        try:
            cleaned_text.encode(terminal_encoding, errors="strict")
            return cleaned_text
        except UnicodeEncodeError:
            # 替换无法编码的字符 Replace unencodable characters
            return cleaned_text.encode(terminal_encoding, errors="replace").decode(terminal_encoding)

    # 其他类型转为字符串后处理 Processing after converting other types to strings
    else:
        return safe_decode(str(text), fallback_encoding)
