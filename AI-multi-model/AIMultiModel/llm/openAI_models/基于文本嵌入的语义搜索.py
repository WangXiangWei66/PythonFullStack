# ast（Abstract Syntax Trees，
# 抽象语法树）模块主要用于处理 Python 代码的抽象语法树。
# 抽象语法树是源代码的一种结构化表示，它以树状结构来描述代码的语法结构，
# 每个节点代表一个语法结构单元。
import ast
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(
    # base_url : 指定第三方代理
    base_url="https://api.96open.com/v1",
    # api_key : 为第三方代理的密匙
    api_key='sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB'
)

#给出一个关键词或者一句话，去原始向量中去搜索相似的用户评论
# 创建文本嵌入
def text_to_embedding(text, model):
    return client.embeddings.create(input=text, model=model).data[0].embedding


# cosine : 余弦
def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# literal_eval : 解释或者计算一个字符串,把str变成矩阵
# df中要包含搜索的数据
# search_st : 要搜索的文本字符串
# top_n=3 : 返回三个最相似文本的数量
def search_text(df, search_st, top_n=3):
    df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)
    input_vector = text_to_embedding(search_st, 'text-embedding-3-small')

    df['similarity'] = df.embedding_vec.apply(lambda vec: cosine_distance(vec, input_vector))

    res = (df.sort_values('similarity', ascending=False)
           .head(top_n)
           .text_content.str.replace('Summary:', "")
           .str.replace("; Text:", ';')
           )
    for r in res:
        print(r)
        print(('-' * 30))


if __name__ == '__main__':
    df = pd.read_csv('../datas/output_embedding.csv')

    search_text(df, 'Method of cooking', top_n=5)
