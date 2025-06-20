from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import tool, StructuredTool   #用于创建自定义工具
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field  #用于定义工具输入的结构化模型数据

llm = ChatOpenAI(
    temperature=1.0,
    model='gpt-3.5-turbo',
    api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
    base_url="https://xiaoai.plus/v1"
)

#下面创建了一个用于搜索本地数据的工具
class SearchInput(BaseModel):  #使用Pydantic定义输入模型，要求传入搜索关键词query
    query: str = Field(description='你搜索的关键词')

#@tool装饰器：将函数转换为可被代理调用的工具
#指定工具名为my_search_tool，输入验证模型为SearchInput，return_direct=True表示工具输出直接作为结果返回
@tool('my_search_tool', args_schema=SearchInput, return_direct=True)
def my_search(query: str) -> str:
    '''用来搜索我电脑上的数据'''
    return '我是一个搜索的工具'


def sort_num(num: str):
    '''把所给的数字重新排序'''
    return sorted(eval(num))  #eval将字符串转化为列表

#定义输入模型，要求传入待排序的字符串
class SortToolInput(BaseModel):
    num: str = Field(description='待排序的类别字符串')


# 2、结构化，得到一个工具
sort_tool = StructuredTool.from_function(
    func=sort_num,
    name='sort_num',
    description='排序列表的数字',
    args_schema=SortToolInput,
    return_direct=True
)

# 自带的，加载预定义工具并整合自定义工具
tools = load_tools(['llm-math', 'arxiv'], llm)
tools = [sort_tool, my_search] + tools

#创建一个能够自主选择和调用工具的智能代理
agent = initialize_agent(
    tools,  #代理可使用的工具列表
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

agent.run('计算 2 ** 5 + 4， 并且给`[3, 6, 9, 23, 1,]`排一下序')
