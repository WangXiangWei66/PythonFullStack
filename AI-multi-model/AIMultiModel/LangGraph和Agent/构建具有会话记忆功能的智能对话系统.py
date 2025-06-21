import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver  #内存检查点，用于保存会话状态
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition #处理工具调用节点和条件判断函数
from typing import Annotated
from typing_extensions import TypedDict
from AIMultiModel.LangGraph和Agent.构建可视化图和实现流式对话功能函数 import loop_graph_invoke


# 1、定义一个状态  类型
#使用Annotated类型注解关联add_messages函数，确保消息列表可被自动更新和管理
class MyState(TypedDict):  # 在整个流程图中，状态用来保存历史记录
    messages: Annotated[list, add_messages]


# 2、顶替一个流程图
graph = StateGraph(MyState)

# 3、准备一个node节点，并且把他添加到流程图中
llm = ChatOpenAI(
    temperature=1.0,
    model='gpt-3.5-turbo',
    api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
    base_url="https://xiaoai.plus/v1"
)

# llm = ChatOpenAI(
#     model='glm-4-plus',
#     temperature=1.0,
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     api_key="e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv"
# )
# 4、添加一个工具节点（互联网搜索工具）
os.environ["TAVILY_API_KEY"] = "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"  # 定义Tavily搜索工具的API密钥
search_tool = TavilySearch(max_results=2)  # 创建搜索工具实例并设置最大返回结果数
tools = [search_tool]

agent = llm.bind_tools(tools)

#定义agent节点处理逻辑，该函数会被状态图调用
def chatbot(state: MyState):
    return {'messages': [agent.invoke(state['messages'])]}

#添加agent节点，使用chatbot函数处理用户输入，生成回复或决定是否调用工具
graph.add_node('agent', chatbot)

# 添加一个工具节点
tool_node = (ToolNode(tools=tools))  #封装成列表tools，用于执行实际的工具调用
graph.add_node('tools', tool_node)

# 根据智能体自动决策是否调用工具
graph.add_conditional_edges('agent', tools_condition)

# 设置边
graph.add_edge('tools', 'agent')

# 设置入口节点
graph.set_entry_point('agent')
#创建内存检查点，将会话状态保存在内存中
memory_checkpointer = MemorySaver()
# sqlite_checkpointer = SqliteSaver('sqlite://test.db')
#编译状态图，并关联检查点，使系统能够保存和回复对话历史，实现会话记忆功能
graph = graph.compile(checkpointer=memory_checkpointer)

#启动对话循环
thread_id = input('请输入一个sessionId:')
#将会话ID存入配置，传递给状态图
config = {"configurable": {"thread_id": thread_id}}

while True:
    try:
        user_input = input('用户：')
        if user_input.lower() in ['exit', 'q', 'quit']:
            print('对话结束，拜拜')
            break
        else:
            loop_graph_invoke(graph, user_input, config)
    except Exception as e:
        print(e)
