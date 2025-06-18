# langchain_core:是一个与语言相关的类 message子模块 处理与对话消息相关的逻辑
# SystemMessage用于表示系统级别的消息，例如在多轮对话中系统给出的指令或提示信息
# HumanMessage ： 表示人类用户输入的信息，以便在对话处理逻辑中区分不同来源的消息进行相应的处理
from langchain_core.messages import SystemMessage, HumanMessage
# out_parsers: 用于处理输出解析相关功能
# StrOutputParser:将模型的输出解析为字符串格式
from langchain_core.output_parsers import StrOutputParser
# prompts:用于处理与提示相关的逻辑
# ChatPromptTemplate:用于创建聊天提示模板工具
from langchain_core.prompts import ChatPromptTemplate
# langchain_openai 是 LangChain 与 OpenAI 集成的相关库
# ChatOpenAI:开发者可以调用OpenAI的聊天模型功能
from langchain_openai import ChatOpenAI

llm_glm = ChatOpenAI(
    model='glm-4-plus',
    # temperature越高使文本更具创建性和多样性
    temperature=1.0,
    openai_api_key='e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

llm_deepseek = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    base_url="https://api.deepseek.com",
    api_key='sk-994985da2eb24831a902b05e1ea97088'
)
# prompt:是一种聊天提示模板
# .from_messages方法接受一个包含多个消息元组的列表作为参数 第一个元素是消息的角色标识  第二个元素是消息的内容模板
prompt = ChatPromptTemplate.from_messages(
    [
        # {language} 是一个占位符，在实际使用模板时需要传入具体的目标语言，
        ('system', '请把下面的语句翻译成{language}'),
        ('user', '{user_text}')
    ]
)

parser = StrOutputParser()

# 管道操作符（|）来构建一个处理链 chain
chain = prompt | llm_deepseek | parser
# chain.invoke：用于执行整个链式操作的关键方法
print(chain.invoke({'language': '日文', 'user_text': '今天天气怎么样？'}))

#configurevt. 安装；使成形