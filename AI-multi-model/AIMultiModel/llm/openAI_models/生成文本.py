#os 模块，你可以使用它里面的函数和属性来执行各种操作系统相关的任务。
import os
from openai import OpenAI

client = OpenAI(
    # base_url : 指定第三方代理
    base_url="https://api.96open.com/v1",
    # api_key : 为第三方代理的密匙
    api_key='sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB'
)
#completion ：指的是文本生成任务
#client.chat.completions:调用了 OpenAI 客户端（client）的 chat.completions.create 方法来创建一个文本生成请求。
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=1,
    #messages 列表定义了与模型交互的对话内容。
    messages=[
        #“system” 角色设定了模型的行为背景，这里将模型设定为 “快乐的小助手”，引导模型以这样的角色风格进行回复。
        {"role": "system", "content": "你是一个快乐的小助手"},
        #“user” 角色代表用户的输入
        {"role": "user", "content": "帮我生成一个200字的：参加家长会议的心得体会。"}
    ]
)
#它会展示整个生成结果对象的详细信息，包括生成过程中的各种元数据，如使用的模型、请求 ID、响应头信息等，以及生成的文本内容相关信息。
print(completion)
print(completion.choices[0].message)
print(completion.choices[0].message.content)
