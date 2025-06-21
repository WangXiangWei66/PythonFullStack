'''本代码构建了一个融合AI自动处理、工具调用和人工介入的智能对话系统'''
import os
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from AIMultiModel.LangGraph和Agent.构建可视化图和实现流式对话功能函数 import draw_graph


class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    ask_person: bool  # 决定是否请求人工接入标签


def create_response(response: str, ai_message: AIMessage):
    """
    创建一个ToolMessage，包含响应内容和工具调用ID。
    参数:
    - response (str): 响应文本内容。
    - ai_message (AIMessage): AI消息对象，从中获取工具调用的ID。
    返回:
    - ToolMessage: 包含响应内容和工具调用ID的消息对象。
    """
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

#触发人工客服介入的工具，包含用户请求内容
class AskPersonMessage(BaseMessage):
    """
    将对话升级到人工客服。
    如果您无法直接提供帮助，或者用户需要超出您权限的支持，请使用此功能。
    要使用此功能，请转达用户的“请求”，以便人工客服可以提供正确的指导。
    """
    request: str  # 普通用户的请求内容，需要传达给人工客服以获得适当的帮助


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
agent = llm.bind_tools(tools + [AskPersonMessage])


# chatbot 节点函数如何以当前 State 作为输入，并返回一个包含更新后的 messages
# 第一个参数是唯一的节点名称
# 第二个参数是每当节点被使用时将调用的函数或对象
def chatbot(state: MyState):
    resp = agent.invoke(state['messages'])
    ask_person = False  # 默认概况下，不需要人工客服
    # 判断一：响应中包含工具调用， && 判断二：工具调用的名字与RequestAssistance类名匹配
    if resp.tool_calls and resp.tool_calls[0]['name'] == AskPersonMessage.__name__:
        ask_person = True
    return {'messages': [resp], "ask_person": ask_person}


graph.add_node('agent', chatbot)

# 添加一个工具节点
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)


# 增加一个人工介入的节点
def person_node(state: MyState):
    """
    处理需要人工客服介入的节点逻辑。
    :param state:
    :return:
    """
    input_answer = input('人工给一个答案：')

    # 创建一个消息
    new_messages = []
    if not isinstance(state['messages'][-1], ToolMessage):
        new_messages.append(input_answer, state['messages'][-1])

    return {
        'messages': new_messages,
        'ask_person': False,
    }


graph.add_node('person', person_node)


def select_next_node(state: MyState):
    """
        根据当前状态选择下一个节点。
        参数:
        - state (State): 当前的状态字典，包含所有消息及标志位。
        返回:
        - str: 下一个要执行的节点名称。
        """
    # 如果当前状态指示需要人类介入，则返回"human"节点
    if state["ask_person"]:
        return "person"

    # 否则，我们可以像以前一样路由到其他节点
    # 这里使用了`tools_condition`函数来决定下一步
    return tools_condition(state)


# 根据智能体自动决策是否需要调用工具
graph.add_conditional_edges(
    'agent',
    select_next_node,
    {  # 路由匹配
        'person': 'person',
        'tools': 'tools',
        END: END
    })

# 设置边
graph.add_edge('tools', 'agent')
graph.add_edge('person', 'agent')
# 设置入口节点
graph.set_entry_point('agent')
memory_checkpointer = MemorySaver()

# 确定整个graph
graph = graph.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=['person']
)

# 把graph变成一张图
draw_graph(graph, 'graph5.png')
