'''本代码实现的功能：
1、接受用户对餐厅的偏好程度
2、将用户偏好总结为清晰的需求
3、基于总结的需求推荐3家餐厅并给出理由
4、对推荐内容机型精简总结，方便用户快速浏览'''
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,  #使得模型的输入为完全随机的
    model='deepseek-chat',
    base_url="https://api.deepseek.com",
    api_key='sk-994985da2eb24831a902b05e1ea97088'
)

gather_preferences_prompt = ChatPromptTemplate.from_template(
    "用户输入了一些餐厅偏好：{input1}\n"
    "请将用户的偏好总结为清晰的需求："
)

recommend_restaurants_prompt = ChatPromptTemplate.from_template(
    "基于用户需求：{input2}\n"
    "请推荐 3 家适合的餐厅，并说明推荐理由："
)

# 下面总结推荐内容供用户快速参考

summarize_recommendations_prompt = ChatPromptTemplate.from_template(
    "以下是餐厅推荐和推荐理由：\n{input3}\n"
    "请总结成 2-3 句话，供用户快速参考："
)

chain = gather_preferences_prompt | llm | recommend_restaurants_prompt | llm | summarize_recommendations_prompt | llm | StrOutputParser()

print(chain.invoke({'input1': '我喜欢安静的地方， 有素食的餐厅更好，而且价格也不贵。'}))
