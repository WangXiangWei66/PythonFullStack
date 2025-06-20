'''本代码的功能：
1、初始化一个大语言模型作为核心推理引擎
2、加载了两个工具，llm-math用于数学计算，arvix用于查询学术论文
3、创建了一个能够理解和使用工具的智能代理
4、演示了如何使用该代理查询ArXiv论文的创新点'''

#initialize_agent: 用于创建能够使用工具的智能代理，
# AgentType: 定义代理的类型，这里使用的是基于聊天的零样本学习类型React框架
from langchain.agents import initialize_agent, AgentType
#load_tools: 加载预定义的工具集合
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model='glm-4-plus',
    # temperature越高使文本更具创建性和多样性
    temperature=1.0,
    openai_api_key='e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

tools = load_tools(['llm-math', 'arxiv'], llm)

agent = initialize_agent(
    tools,  #代理可以使用的工具列表
    llm,  #核心的代理推理引擎
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, #聊天场景优化的零样本学习代理，使用 ReAct 框架
    handle_parsing_errors=True,  #当解析工具输出出错时，尝试继续执行而不是中断
    verbose=True  #启用详细的输出模式，显示代理的思考过程和工具调用情况
)

# resp = agent.invoke('What is the 25% of 300')
# print(resp)

resp = agent.invoke('介绍一下2006.13145这篇论文的创新点？')
print(resp)