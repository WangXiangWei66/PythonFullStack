import os
from typing import  Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition  #处理当前节点和条件判断数
from AIMultiModel.LangGraph和Agent.构建可视化图和实现流式对话功能函数 import loop_graph_invoke, loop_graph_invoke_tools,draw_graph


# 1、定义了一个状态类型
#定义对话过程中需要维护的状态数据结构
class MyState(TypedDict):  # 在整个流程图中，状态用来保存历史记录
    messages: Annotated[list, add_messages]

#初始化图和大语言模型
# 2、定义一个流程图
graph = StateGraph(MyState) #创建一个MyState类型的状态图，用于管理整个对话流程

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
#配置和绑定搜索工具
os.environ["TAVILY_API_KEY"] = "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"  # 定义Tavily搜索工具的API密钥
search_tool = TavilySearch(max_results=2)  # 创建搜索工具实例并设置最大返回结果数
tools = [search_tool]
# 将工具和大模型绑定
agent = llm.bind_tools(tools)


# chatbot 节点函数如何以当前 State 作为输入，并返回一个包含更新后的 messages
# 第一个参数是唯一的节点名称
# 第二个参数是每当节点被使用时将调用的函数或对象

#定义对话处理节点数
def chatbot(state: MyState):
    return {'messages': [agent.invoke(state['messages'])]}


graph.add_node('agent', chatbot)

# 添加工具节点
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

# 根据智能体自动决策是否需要调用工具
graph.add_conditional_edges('agent', tools_condition)

# 设置边
graph.add_edge('tools', 'agent')

# 设置入口节点
graph.set_entry_point('agent')
# 我们正在使用内存中的检查点。它将所有内容都保存在内存中。
# 在生产应用程序中，您可能会将其更改为使用 SqliteSaver 或 PostgresSaver 并连接到您自己的数据库
# pip install langgraph-checkpoint-sqlite
#配置检查点，并编译状态图
memory_checkpointer = MemorySaver()
# sqlite_checkpointer = SqliteSaver('sqlite:///test.db')

# 整个graph已经确定
graph = graph.compile(
    checkpointer=memory_checkpointer,
    # interrupt_before=['tools']  ## 在执行到"tools"节点之前中断，允许外部处理或检查
    # 注意：如果需要，也可以在工具执行后中断
    interrupt_after=["tools"]
)

# draw_graph(graph, 'graph4.png')
#启动对话循环
thread_id = input('请输入一个sessionId：')
config = {'configurable': {'thread_id': thread_id}}

# 执行工作流
while True:
    try:
        user_input = input('用户：')
        if user_input.lower() in ['q', 'quit', 'exit']:
            print('对话结束，拜拜')
            break
        else:
            # 执行工作流
            loop_graph_invoke(graph, user_input, config)

            # 查看此时的状态
            now_state = graph.get_state(config)
            # print('此时的状态数据：', now_state)

            if 'tools' in now_state.next:
                tools_script_messages = now_state.values['messages'][-1]
                print('Tool Script:', tools_script_messages.tool_calls)

                if input('人工智能输入是否继续工具调用：yes 或者 no').lower() == 'yes':
                    loop_graph_invoke_tools(graph, None, config)

    except Exception as e:
        print(e)
