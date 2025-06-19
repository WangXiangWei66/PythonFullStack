'''本代码实现的功能：
1、通过第一个提示词模板生成指定主题的短文
2、将生成的短文传递给第二个提示词模板进行评分和评价
3、展示了两种不同的链式调用方式，并通过自定义函数打印中间结果'''



from langchain_core.output_parsers import StrOutputParser #将大语言模型输出转化为字符串
from langchain_core.prompts import ChatPromptTemplate #创建提示词模板
from langchain_core.runnables import RunnableLambda #封装自定义函数为可运行组件
from langchain_openai import ChatOpenAI #langchain提供了OPenAI聊天模型接口

llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    base_url="https://api.deepseek.com",
    api_key='sk-994985da2eb24831a902b05e1ea97088'
)

prompt1 = ChatPromptTemplate.from_template('给我写一篇关于{key_word}的{type}，字数不超过{count}。')

prompt2 = ChatPromptTemplate.from_template('请简单评价一下这篇短文，如果总分是10分，请给这篇短文打分： {text_content}')

chain1 = prompt1 | llm | StrOutputParser()
#方法一
# chain2 = {'text_content' : chain1} |prompt2 | llm | StrOutputParser()

#方法2
def print_chain1(input):
    print(input)
    print('--' * 30)
    return {'text_content': input}

chain2 = chain1 | RunnableLambda(print_chain1) | prompt2 | llm | StrOutputParser()
print(chain2.invoke({'key_word' : '青春','type' : '散文','count' : 400}))
