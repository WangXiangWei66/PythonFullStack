from typing import Annotated  #为类型添加额外的元数据或者注解

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END #预定义的特殊节点，表示流程的开始和结束
#StateGraph：LangGraph 的核心类，用于定义和管理状态转换图。
from langgraph.graph import add_messages, StateGraph
from mypy.build import TypedDict  #定义状态的类型结构

#状态、节点、边

# 1、定义一个状态  类型
class MyState(TypedDict):  # 在整个循环过程中，状态用来保存历史记录
    #messages:状态中保存数据的key
    #list为数据类型，add_messages更新list数据
    messages: Annotated[list, add_messages]


# 2、定义一个流程图
graph = StateGraph(MyState)

# 3、准备一个node，并且把他添加到流程图中
llm = ChatOpenAI(
    temperature=1.0,
    model='gpt-3.5-turbo',
    api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
    base_url="https://xiaoai.plus/v1"
)


def chatbot(state: MyState):
    return {'messages': [llm.invoke(state['messages'])]}

# chatbot 节点函数如何以当前 State 作为输入，并返回一个包含更新后的 messages
# 第一个参数是唯一的节点名称
# 第二个参数是每当节点被使用时将调用的函数或对象
graph.add_node('chatbot', chatbot)

# 4、设置边
graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)
#编译状态图，生成可执行的流程
graph = graph.compile()  # 构建一个图对象


# 5、把graph变成一张图
# try:
#     image = graph.get_graph().draw_mermaid_png()
#     with open('graph.png', 'wb') as f:
#         f.write(image)
# except Exception as e:
#     print(e)
#封装图的执行逻辑，将用户输入转换为状态，调用图处理并输出 AI 回复。
def loop_graph_invoke(user_input: str):
    '''循环调用这个流程图，让AI可以一直和用户对话'''
    for chunk in graph.stream({'messages': [('user', user_input)]}):
        for value in chunk.values():
            print('AI机器人:', value['messages'][-1].content)


while True:
    try:
        user_input = input('用户：')
        if user_input.lower() in ['q', 'quit', 'exit']:
            print('对话结束，拜拜！')
            break
        else:
            loop_graph_invoke(user_input)
    except Exception as e:
        print(e)
