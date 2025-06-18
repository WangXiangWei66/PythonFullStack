# SQLChatMessageHistory:用于处理基于 SQL 数据库来存储和管理聊天消息历史记录的功能
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
# prompts：处理与提示相关的逻辑
# ChatPromptTemplate 用于创建聊天场景下的提示模板
#MessagePlaceholder：作为占位符，在提示模板中插入对话历史等消息，方便在聊天过程中整合历史信息进行连贯的交互
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# runables：包含与可运行对象相关的功能
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
#sqlalchemy:python Sql包和对象关系映射库
from sqlalchemy import create_engine  # 新增导入

# 创建一个ChatOpenAi类的一个实例
llm_openai = ChatOpenAI(
    temperature=1.0,
    model='gpt-4o',
    base_url="https://api.96open.com/v1",
    # api_key : 为第三方代理的密匙
    api_key='sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB'
)

# 上下文交互：保存历史纪录（存在哪里）
# StrOutputparser:将模型的输出转化为字符串格式，自动从对象解析需要的文本
parser = StrOutputParser()
#ChatPromptTemplate.from_messages（直接接二元组即可）
#'humans':用户向大语言模型提出的问题
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个幽默的聊天机器人'),
    # MessagePlaceholder:用于在提示中插入对话历史，引入的对话历史变量名为history
    MessagesPlaceholder(variable_name='history'), #表示保存所有聊天记录
    ('human', '{input}')
])
# 提示词|大模型|解析     谁先就收的参数，谁的链放在前面
chain = prompt | llm_openai | parser

# 创建数据库连接引擎（新增代码）
db_engine = create_engine('sqlite:///history.db')

# 将聊天记录保存在本地数据库中（修改后的函数）
def get_session_history(sid):
    return SQLChatMessageHistory(
        session_id=sid,
        connection=db_engine  # 使用 connection 参数替代 connection_string
    )

# RunnableWithMessageHistory :处理带有消息历史记录的可运行任务
runnable = RunnableWithMessageHistory(
    chain,
    get_session_history, #根据会话的ID读取和保存历史记录，必须要BaseChatMessageHistory
    input_messages_key='input',
    history_messages_key='history'
)
#configurable:可配置的、结构的
#session_id不同会有不同的会话，相同才会有上下文的交互
res1 = runnable.invoke({'input': '中国一共有哪些直辖市？'}, config={'configurable': {'session_id': 'laoX002'}})
print(res1)
print('--' * 30)
res2 = runnable.invoke({'input': '这些城市中，那个面积最大？'}, config={'configurable': {'session_id': 'laoX002'}})
print(res2)