from openai import OpenAI#OpenAi： 用于创建API客户端实例
#客户，委托人；客户机，用户端
#初始化client，如果不翻墙要传中间商的地址，翻墙的话可以不传base_url
#如果直接连openai的官网，只传api_key即可
client = OpenAI(
    #base_url : 指定第三方代理
    base_url="https://api.96open.com/v1",
    #api_key : 为第三方代理的密匙
    api_key='sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB'
)

#直接连openai官网
# client = OpenAI(
#     base_url="https://xiaoai.plus/v1",
#     api_key='sk-doD81WgxSoF9A6xYzhgW7GUh5frRwPETI8mDq3ce4UaWnCPF'
# )
#调用嵌入模型接口，生成输入文本的向量表示
resp = client.embeddings.create(
    #model ： 指定使用的嵌入模型名称
    model = 'text-embedding-3-large',
    # 需要生成的嵌入向量的文本
    input = '我喜欢AI大模型开发',
    #OpenAI 嵌入模型（如 text-embedding-3-large）的输出维度由模型本身决定（该模型默认维度为 1536）
    dimensions = 512#使用他可以进行降维
)
#resp.data ： 响应中包含的嵌入数据列表
#data[0]：获取列表中的第一个元素（对应输入文本的嵌入向量）。
print(resp.data[0].embedding)  # 打印生成的向量，在-1到1之间
print(len(resp.data[0].embedding))  # 输出向量长度