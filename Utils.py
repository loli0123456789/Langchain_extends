import os
from langchain_openai import ChatOpenAI


def loadEnv():
    # 加载 .env 到环境变量
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())