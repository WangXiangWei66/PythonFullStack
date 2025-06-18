from openai import OpenAI

client = OpenAI(
    base_url = "https://api.deepseek.com",
    api_key = "sk-994985da2eb24831a902b05e1ea97088"
)

completion = client.chat.completions.create(
    model = "deepseek-chat",
    messages=[
        {"role":"system","content":"你是一个聪明且富有创造力的小说作家"},
        {"role":"user","content":"请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"}
    ],
    # top_p 参数设置为 0.7，它是一种核采样参数，用于控制生成文本的随机性与多样性，值越小生成越保守。
    # temperature 参数设置为 0.9，它同样影响生成文本的随机性，值越高生成结果越具多样性和创造性。
    top_p=0.7,
    temperature=0.9
)

print(completion)
print(completion.choices[0].message)
print(completion.choices[0].message.content)
