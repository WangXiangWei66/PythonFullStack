from openai import OpenAI

client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key="e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv"
)

completion = client.chat.completions.create(
    model="glm-4-plus",#该参数价格较低
    messages=[
        {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
        {"role": "user",
         "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"}
    ],
    # top_p 参数设置为 0.7，它是一种核采样参数，用于控制生成文本的随机性与多样性，值越小生成越保守。
    # temperature 参数设置为 0.9，它同样影响生成文本的随机性，值越高生成结果越具多样性和创造性。
    top_p=0.7,
    temperature=0.9
)

print(completion)
print(completion.choices[0].message)
print(completion.choices[0].message.content)
