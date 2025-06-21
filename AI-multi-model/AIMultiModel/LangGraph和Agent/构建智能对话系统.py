import os

from langchain_tavily import TavilySearch  #用于实现互联网搜索
from langchain_openai import ChatOpenAI
#StateGraph构建状态图的核心类
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition  #处理工具调用的节点和条件函数
from typing import Annotated
from AIMultiModel.LangGraph和Agent.构建可视化图和实现流式对话功能函数 import draw_graph, loop_graph_invoke
from typing_extensions import TypedDict


# 需求：对话的案例
# 1、定义一个状态，类型
#定义对话过程中需要维护的状态数据结构
#messages：存储对话历史的列表
#使用Annotated类型注解关联add_messages函数，确保消息列表可被自动更新。
class MyState(TypedDict):
    messages: Annotated[list, add_messages]


# 2、定义流程图

graph = StateGraph(MyState)

# 3、准备一个node节点，并且把他添加到流程图中

# llm = ChatOpenAI(
#     temperature=1.0,
#     model='gpt-4o',
#     api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
#     base_url="https://xiaoai.plus/v1"
# )

# deepseek - v3可以用，但是会出现死循环

llm = ChatOpenAI(
    model='glm-4-plus',
    temperature=1.0,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key="e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv"
)

# 4、添加一个工具节点（互联网搜索工具）
os.environ["TAVILY_API_KEY"] = "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"  #定义Tavily搜索工具的API密钥
search_tool = TavilySearch(max_results=2)  #创建搜索工具实例并设置最大返回结果数
tools = [search_tool]

# 把工具和大模型绑定
agent = llm.bind_tools(tools)


# chatbot节点函数如何以当前的State作为输入，并返回一个包含更新后的messges
# 第一个参数是唯一的节点名称
# 第二个参数是每当节点被使用时将调用的函数或对象
def chatbot(state: MyState):
    return {'messages': [agent.invoke(state['messages'])]}

#向状态图中添加工具节点
graph.add_node('agent', chatbot)  #使用chatbot函数处理用户输入，生成回复或决定是否调用工具
# 添加一个工具节点
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

# 根据智能体自动决策是否需要调用工具
graph.add_conditional_edges('agent', tools_condition)

# 5、设置边
graph.add_edge('tools', 'agent')
# 设置入口节点
graph.set_entry_point('agent')

# 确定整个graph
graph = graph.compile()  # 构建图对象

# 6、把graph变成一张图
# draw_graph(graph, 'graph2.png')

# 执行工作流
while True:
    try:
        user_input = input('用户：')
        if user_input.lower() in ['quit', 'q', 'exit']:
            print('对话结束，拜拜')
            break
        else:
            loop_graph_invoke(graph, user_input)
    except Exception as e:
        print(e)
