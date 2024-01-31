from llms.ZhipuLLM import Zhipu
from Utils import loadEnv

loadEnv()

llm=Zhipu()

result=llm.invoke("你是谁？")
print(result)