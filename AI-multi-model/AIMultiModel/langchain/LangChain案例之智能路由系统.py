'''本代码构建了一个智能路由系统：
1、接收用户在不同领域的问题
2、通过第一个LLM调用判断问题所属领域
3、根据领域动态选择对应的提示词模板和处理链
4、使用特定的提示词模板生成最终的回答'''
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RouterRunnable, RunnableSequence, router, RunnableLambda  #RunnableSequence用于创建顺序执行的组件连
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,  # 使得模型的输入为完全随机的
    model='deepseek-chat',
    base_url="https://api.deepseek.com",
    api_key='sk-994985da2eb24831a902b05e1ea97088'
)

physics_template = ChatPromptTemplate.from_template(
    "你是一位物理学教授，擅长用简洁易懂的方式回答物理问题。以下是问题内容：{input}"
)

math_template = ChatPromptTemplate.from_template(
    "你是一位数学家，擅长分步骤解决数学问题，并提供详细的解决过程。以下是问题内容：{input}"
)

history_template = ChatPromptTemplate.from_template(
    "你是一位历史学家，对历史事件和背景有深入研究。以下是问题内容：{input}"
)

computerScience_template = ChatPromptTemplate.from_template(
    "你是一位计算机科学专家，擅长算法、数据结构和编程问题。以下是问题内容：{input}"
)

default_template = ChatPromptTemplate.from_template(
    "输入内容无法归类，请直接回答：{input}"
)

#将用户问题与特定领域的提示词结合调用LLM
default_chain = default_template | llm
physics_chain = physics_template | llm
math_chain = math_template | llm
history_chain = history_template | llm
computerScience_chain = computerScience_template | llm


# 动态路由的chain
def route(input):
    '''根据大模型第一次处理时的输出来，动态判断各种领域的任务'''
    if '物理' in input['type']:
        print('1号')
        return {"key": 'physics', 'input': input['input']}
    elif '数学' in input['type']:
        print('2号')
        return {"key": 'math', 'input': input['input']}
    elif '历史' in input['type']:
        print('3号')
        return {"key": 'history', 'input': input['input']}
    elif '计算机' in input['type']:
        print('4号')
        return {"key": 'computerScience', 'input': input['input']}
    else:
        print('5号')
        return {"key": 'default', 'input': input['input']}


# 创建路由节点
route_runnable = RunnableLambda(route)   #根据路由键选择对应的处理链

# 路由调度器
router = RouterRunnable(runnables={
    'physics': physics_chain,
    'math': math_chain,
    'history': history_chain,
    'computerScience': computerScience_chain,
    'default': default_chain,
})

# 第一个提示词模板
first_prompt = ChatPromptTemplate.from_template(
    "不要回答下面用户的问题，只要根据用户的输入来判断分类，一共有[物理，历史，计算机，数学，其他]5种类别。\n\n \
        用户的输入：{input} \n\n \
        最后的输出包含分类的类别和用户输入的内容，输出格式为json. 其中，类别的key为type，用户输入内容的key为input"
)

chain1 = first_prompt | llm | JsonOutputParser()

chain2 = RunnableSequence(chain1, route_runnable, router, StrOutputParser())

inputs = [
    {"input": "什么是黑体辐射？"},  # 物理问题
    {"input": "计算 2 + 2 的结果。"},  # 数学问题
    {"input": "介绍一次世界大战的背景。"},  # 历史问题
    {"input": "如何实现快速排序算法？"}  # 计算机科学问题
]

for inp in inputs:
    result = chain2.invoke(inp)
    print(f'问题: {inp} \n 回答: {result} \n')