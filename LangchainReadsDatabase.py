# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/15 11:38
# FileName: LangchainReadsDatabase
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
"""
可以使用chain读取数据库，也可以使用代理读取数据库，但是通过代理会复杂一些，因为需要使用API循环读取数据库
优点：大语言模型会根据用户的提问自动生成sql语句进行查询，自动生成想要的答案

One can either use the chain to access the database or use a proxy to do so. However, using a proxy is a bit more
complicated as it requires using an API to loop through and access the database. Advantages: The large language model
will automatically generate SQL statements based on the user's question for querying, and generate the desired
answers automatically.
"""
import os
import sys
import importlib

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

key_path = r"D:\Resource\keys\api_keys.py"
sys.path.append(os.path.dirname(key_path))
api_keys_name = os.path.basename(key_path).split('.')[0]
api_keys = importlib.import_module(api_keys_name)

"""
In Langchain, when using SQLAlchemy for connection, 
SQLAlchemy uses the function `create_engine(r"sqlite:///C:\path\...\xx.db")` to generate a database engine object from the URL.
"""
# URL format: dialect (database) + driver://username:password@hostname:port/database name

# (1)SQLite database
engine1 = create_engine(r"sqlite:////absolute/path/.../xxx.db")  # Unix/Mac
engine2 = create_engine("sqlite:///C:\\path\\...\\xxx.db")  # Windows
engine3 = create_engine(r"sqlite:///C:\path\...\xxx.db")  # Windows

# (2)MySQL database
# default，A connector is required, but its use is not recommended.
# engine4 = create_engine("mysql://scrott:tiger@localhost/foo?charset=utf8")

# mysqlclient,It needs to be installed (pip install mysqlclient), and it is recommended to use it.
# engine5 = create_engine("mysql+mysqldb://scrott:tiger@localhost/foo?charset=utf8")

# pymysql,It needs to be installed (pip install pymysql), and it is recommended to use it.
# engine6 = create_engine("mysql+pymysql://scrott:tiger@localhost/foo?charset=utf8")

username = api_keys.username
pw_mysql = api_keys.pw_mysql
port = '3306'
hostname = "127.0.0.1"
db_name = "test_db1"

# pymysql driver URL
mysql_url = f"mysql+pymysql://{username}:{pw_mysql}@{hostname}:{port}/{db_name}?charset=utf8"

# Create a database by langchain
db = SQLDatabase.from_uri(mysql_url)

# Test whether the database connection is successful and query useful tables.
print(db.get_usable_table_names())

# Query a certain table
print(db.run("select * from users_langchain"))
# mysql> select * from users_langchain;
# +----+-------+--------+-----+-----------------------+-----------------------------------+
# | id | name  | gender | age | job                   | hobby                             |
# +----+-------+--------+-----+-----------------------+-----------------------------------+
# |  1 | jason | 男     |  34 | RPA开发工程师         | 旅游，越野创越                    |
# |  2 | jack  | 男     |  33 | 销售经理              | 打蓝球                            |
# |  3 | awen  | 男     |  27 | 后端开发工程师        | 炒股                              |
# |  4 | hda   | 女     |  23 | 主播                  | 直播                              |
# |  5 | dhudh | 女     |  26 | 户外博主              | 穿越无人区，机车，驾驶            |
# +----+-------+--------+-----+-----------------------+-----------------------------------+
# Result as the follow:
# ['users_langchain']
# [(1, 'jason', '男', 34, 'RPA开发工程师', '旅游，越野创越'), (2, 'jack', '男', 33, '销售经理', '打蓝球'), (3, 'awen', '男', 27, '后端开发工程师',
# '炒股'), (4, 'hda', '女', 23, '主播', '直播'), (5, 'dhudh', '女', 26, '户外博主', '穿越无人区，机车，驾驶')]
