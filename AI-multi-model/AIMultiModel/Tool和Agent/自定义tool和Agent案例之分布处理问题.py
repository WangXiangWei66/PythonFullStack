from langchain import hub  # 用于从 LangChain Hub 拉取预定义的 Prompt 模板。
# create_structured_chat_agent：创建支持结构化工具调用的聊天代理。
# AgentExecutor：代理执行器，协调代理与工具的交互。
from langchain.agents import create_structured_chat_agent, AgentExecutor
# 确保输入输出格式规范
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel

llm = ChatOpenAI(
    temperature=1.0,
    model='gpt-3.5-turbo',
    api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
    base_url="https://xiaoai.plus/v1"
)


# 使用Pydantic定义工具输入的结构化模型
class ArgsInput(BaseModel):
    a: str = Field(description='第一个字符串')
    b: str = Field(description='第二个字符串')


def count_str(a: str, b: str) -> int:
    return len(a) + len(b)


# 结构化得到一个工具
len_add = StructuredTool.from_function(
    func=count_str,  # 指定工具对应的函数
    name='my_Calculator',  # 工具的名称，要唯一
    description='计算字符串长度的累加和',  # 工具功能的描述，帮助代理决定
    args_schema=ArgsInput,  # 使用定义的Pydantic模型验证输入参数
    return_direct=False  # 工具输出不直接返回给用户，而是由代理进一步处理
)

tools = [len_add]  #将自定义工具放入工具列表
prompt = hub.pull('hwchase17/structured-chat-agent')
agent = create_structured_chat_agent(llm, tools, prompt)

# 初始化：agent的执行器
#初始化代理执行器，负责管理代理与工具的交互路程
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
resp = agent_executor.invoke({'input': '`爱国者导弹拦截`的字符串长度加上`abc`字符串的长度是多少？ langsmith是什么？'})
print(resp)
