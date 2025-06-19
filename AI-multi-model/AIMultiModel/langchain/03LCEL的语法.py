# RunnableLambda:将函数封装成一个可运行的组件
# RunableParella：并行调用时的包装key为输入,value为输出
# RunnablePassThrough：用于处理数据的传递和转换
import time #用于时间相关操作，比如休眠
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tracers import Run  #用于跟踪运行状态，支持生命周期管理


def tool1(x: int):
    return x + 10

#RunnableLambda 允许将普通 Python 函数转换为可链式调用的组件
r1 = RunnableLambda(tool1)


# res = r1.invoke(4)
# batch方法用于批量处理数据
# res = r1.batch([4, 5])

# def test2(prompt: str):
#     # 将输入的字符串按照空格进行分割
#     for item in prompt.split(' '):
#         # 使用yield关键字逐个返回分割后的子字符串
#         yield item

def tool2(prompt: str):
    # 将输入的字符串按照空格进行分割
    return prompt.split(' ')  # 返回列表而不是使用yield生成器

r1 = RunnableLambda(tool2)
res = r1.stream('This is a dog.') #流式处理，逐个返回结果
for chunk in res:
    print(chunk)

r1 = RunnableLambda(tool1)
r2 = RunnableLambda(lambda x: x * 2)

chain1 = r1 | r2  # 串行
print(chain1.invoke(2))

chain = RunnableParallel(r1=r1, r2=r2) #创建并行运行的链，多个组件同时处理数据

# max_concurrency:最大并发数
print(chain.invoke(2, config={'max_concurrency': 3}))

new_chain = chain1 | chain
new_chain.get_graph().print_ascii()  # 打印链的图像描述
print(new_chain.invoke(2))

r1 = RunnableLambda(lambda x: {'key1': x})
r2 = RunnableLambda(lambda x: x['key1'] + 10)
r3 = RunnableLambda(lambda x: x['new_key']['key2'])
#添加新的键值对到字典
# chain = r1 |RunnablePassthrough.assign(new_key = r2)
# chain = r1 |RunnablePassthrough() | RunnablePassthrough.assign(new_key=r2)
# chain = r1 | RunnableParallel(foo = RunnablePassthrough(),new_key = RunnablePassthrough.assign(key2=r2))
chain = r1 | RunnableParallel(foo=RunnablePassthrough(),
                              new_key=RunnablePassthrough.assign(key2=r2)) | RunnablePassthrough().pick(
    ['new_key']) | r3
print(chain.invoke(2))

r1 = RunnableLambda(tool1)
r2 = RunnableLambda(lambda x: int(x) + 20)
# r1报错的情况下，r2是r1的后被选项
chain = r1.with_fallbacks([r2])
print(chain.invoke('2'))

counter = -1


def tool3(x):
    global counter
    counter += 1
    print(f'执行了{counter}次')
    return x / counter


r1 = RunnableLambda(tool3).with_retry(stop_after_attempt=4)
print(r1.invoke(2))

# 根据条件、动态的构建链
r1 = RunnableLambda(tool1)
r2 = RunnableLambda(lambda x: [x] * 2)
# 判断本身也是一个节点
chain = r1 | RunnableLambda(lambda x: r2 if x > 12 else RunnableLambda(lambda x: x))
print(chain.invoke(3))


# 生命周期管理
def tool4(n: int):
    time.sleep(n)
    return n * 2


r1 = RunnableLambda(tool4)


def on_start(run_obj: Run):
    print('r1启动的时间：', run_obj.start_time)


def on_end(run_obj: Run):
    print('r1结束的时间:', run_obj.end_time)

#添加生命周期回调函数
chain = r1.with_listeners(on_start=on_start, on_end=on_end)
print(chain.invoke(2))
