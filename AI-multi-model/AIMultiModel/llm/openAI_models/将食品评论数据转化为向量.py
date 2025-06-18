import pandas as pd
from openai import OpenAI

# console : 控制台，操纵台
# index_col : 指定那一列作为dataframe的行索引
df = pd.read_csv('../datas/fine_food_reviews_1k.csv', index_col=0)

df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
# df.dropna : 用于移除 DataFrame 中包含缺失值（NaN 或 None）的行或列
df = df.dropna()
# str.strip : 去除字符串开头和结尾处的指定字符，默认情况下会去除空白字符，像空格、制表符、换行符等都属于空白字符
df['text_content'] = 'Summary:' + df.Summary.str.strip() + "; Text:" + df.Text.str.strip()

print(df.head(2))

client = OpenAI(
    # base_url : 指定第三方代理
    base_url="https://api.96open.com/v1",
    # api_key : 为第三方代理的密匙
    api_key='sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB'
)


# text_to_embedding : 将文本转化为向量形式
def text_to_embedding(text, model):
    return client.embeddings.create(input=text, model=model).data[0].embedding


# apply:会对text_context 列中的每个元素执行lambda函数
# lamdba ：  是一种用于创建匿名函数的特殊语法
df['embedding'] = df.text_content.apply(lambda x: text_to_embedding(x, model='text-embedding-3-small'))
df.to_csv('datas/output_embedding.csv')

print(df['embedding'][0])
