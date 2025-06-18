# RunnableLambda:将函数封装成一个组件
# RunableParella：并行调用时的包装key为输入,value为输出
# RunnablePassThrough：把输出的非字典的单个值变成字典、把输出的新字典添加新的键或者把他们过滤
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


def test1(x: int):
    return x + 10


r1 = RunnableLambda(test1)


# res = r1.invoke(4)
# batch方法用于批量处理数据
# res = r1.batch([4, 5])

def test2(prompt: str):
    # 将输入的字符串按照空格进行分割
    for item in prompt.split(' '):
        # 使用yield关键字逐个返回分割后的子字符串
        yield item


r1 = RunnableLambda(test2)
res = r1.stream('This is a dog.')
for chunk in res:
    print(chunk)

r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: x * 2)

chain1 = r1 | r2  # 串行
print(chain1.invoke(2))

chain = RunnableParallel(r1=r1, r2=r2)

# max_concurrency:最大并发数
print(chain.invoke(2, config={'max_concurrency': 3}))

new_chain = chain1 | chain
new_chain.get_graph().print_ascii()  # 打印链的图像描述
print(new_chain.invoke(2))

r1 = RunnableLambda(lambda x: {'key1': x})
r2 = RunnableLambda(lambda x: x['key1'] + 10)
r3 = RunnableLambda(lambda x: x['new_key']['key2'])
# chain = r1 |RunnablePassthrough.assign(new_key = r2)
# chain = r1 |RunnablePassthrough() | RunnablePassthrough.assign(new_key=r2)
# chain = r1 | RunnableParallel(foo = RunnablePassthrough(),new_key = RunnablePassthrough.assign(key2=r2))
chain = r1 | RunnableParallel(foo=RunnablePassthrough(),
                              new_key=RunnablePassthrough.assign(key2=r2)) | RunnablePassthrough().pick(
    ['new_key']) | r3
print(chain.invoke(2))

r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: int(x) + 20)
# r1报错的情况下，r2是r1的后被选项
chain = r1.with_fallbacks([r2])
print(chain.invoke('2'))

counter = -1


def test3(x):
    global counter
    counter += 1
    print(f'执行了{counter}次')
    return x / counter


r1 = RunnableLambda(test3).with_retry(stop_after_attempt=4)
print(r1.invoke(2))

# 根据条件、动态的构建链
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: [x] * 2)
# 判断本身也是一个节点
chain = r1 | RunnableLambda(lambda x: r2 if x > 12 else RunnableLambda(lambda x: x))
print(chain.invoke(3))
