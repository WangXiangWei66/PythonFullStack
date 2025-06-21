import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage, AIMessage
from AIMultiModel.LangGraph和Agent.构建可视化图和实现流式对话功能函数 import draw_graph, loop_graph_invoke,loop_graph_invoke_tools   #用于执行对话流程和手动触发工具调用


# 1、定义一个状态  类型
class MyState(TypedDict):
    messages: Annotated[list, add_messages]


# 2、定义一个流程图
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
# 配置和绑定搜索工具
os.environ["TAVILY_API_KEY"] = "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"  # 定义Tavily搜索工具的API密钥
search_tool = TavilySearch(max_results=2)  # 创建搜索工具实例并设置最大返回结果数
tools = [search_tool]
# 将工具和大模型绑定
agent = llm.bind_tools(tools)


# chatbot 节点函数如何以当前 State 作为输入，并返回一个包含更新后的 messages
# 第一个参数是唯一的节点名称
# 第二个参数是每当节点被使用时将调用的函数或对象
def chatbot(state: MyState):
    return {'messages': [agent.invoke(state['messages'])]}


graph.add_node('agent', chatbot)

# 添加一个工具节点
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

# 根据智能体自动决策是否需要调用工具
graph.add_conditional_edges('agent', tools_condition)

# 设置边
graph.add_edge('tools', 'agent')
# 设置入口节点
graph.set_entry_point('agent')
memory_checkpointer = MemorySaver()

# 确定整个graph
graph = graph.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=['tools']
)

# 把graph变成一张图
# draw_graph(graph,'graph5.png')

thread_id = input('请输入一个sessionId:')
config = {"configurable": {"thread_id": thread_id}}

#当用户选择不调用工具时，调用此函数
def get_answer(tool_message):
    '''让人工介入，并且给一个问题的答案'''
    input_answer = input('人工给一个答案：')
    answer = (
        input_answer
    )

    # 创建一个消息
    #人工输入答案后，创建ToolMessage（模拟工具返回结果）和AIMessage（AI 回复）
    new_message = [
        ToolMessage(content=answer, tool_call_id=tool_message.tool_calls[0]['id']),
        AIMessage(content=answer)
    ]

    # 把新人造的信息，添加到工作流的state中
    #使用graph.update_state将人工答案注入对话状态，确保流程继续
    graph.update_state(
        config=config,
        values={'messages': new_message}
    )
    #打印最新消息，显示人工干预的结果
    for msg in graph.get_state(config).values['messages'][-2:]:
        msg.pretty_print()


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
            now_state = graph.get_state(config) #获取当前的组那个太，见擦汗下一个节点是否为tools
            # print('此时的状态数据：', now_state)

            if 'tools' in now_state.next:  # 判断下一个节点是否为：tool
                # 可以人工介入
                tools_script_messages = now_state.values['messages'][-1]  # 状态中存储的最后一个message
                print('Tool Script:', tools_script_messages.tool_calls)

                if input('人工智能输入是否继续工具调用：yes 或者 no').lower() == 'yes':
                    loop_graph_invoke_tools(graph, None, config)
                else:
                    # 用户输入了no,那就需要自己添加一个Message
                    get_answer(tools_script_messages)

    except Exception as e:
        print(e)
