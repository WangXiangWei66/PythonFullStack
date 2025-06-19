'''代码构建了一个基于LangChain和LangGraph的智能问答系统
1、初始化大语言模型和Tavily探索工具
2、创建ReAct代码，经LLM与搜索工具整和
3、同六十接口查询2025年6月19日青岛市的天气信息
4、演示了如何结合LLM的推理能力和外部工具的实时数据获取能力'''
import os  #用户环境变量操作
from langchain_core.messages import HumanMessage #用于用户输入的信息
from langchain_openai import ChatOpenAI #提供的聊天模型接口
from langchain_tavily import TavilySearch  #用于获取实时信息
from langgraph.prebuilt import create_react_agent #LangGraph中用于创建ReAct代理的函数

llm = ChatOpenAI(
    model='glm-4-plus',
    # temperature越高使文本更具创建性和多样性
    temperature=1.0,
    openai_api_key='e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
#通过环境变量设置 Tavily API 密钥，确保搜索工具能正常调用
os.environ["TAVILY_API_KEY"] = "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"
#设置了最大返回的结果
search_tool = TavilySearch(max_results=5, api_key="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")

# res = search_tool.invoke('2025年6月19日，青岛市的天气怎么样')
# print(res)

# 创建一个agent，整合工具
tools = [search_tool] #定义了代理可用的工具列表
#创建ReAct代理，将LLM与工具整合
agent = create_react_agent(llm, tools)

# resp = agent.invoke({'messages': [HumanMessage(content='山东的省会是哪个城市')]})
# resp = agent.invoke({'messages': [HumanMessage(content='2025年6月19日，青岛市的天气怎么样？')]})
# print(resp)

#下面是流式调用代理获取天气信息
for chunk in agent.stream({'messages': [HumanMessage(content='2025年6月19日，青岛市的天气怎么样？')]}):
    print(chunk)
    print('--' * 20)
