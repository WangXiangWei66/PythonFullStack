#需求：用生成器求100以内质数
def odd_num(): #求所有大于1奇数的生成器
    n=1
    while True:
        n=n+2
        yield n

#将能被n整除的数过滤掉
def my_filter(n):
    return lambda x: x % n >0

def tt():
    yield 2#这里会返回一次结果
    g=odd_num()
    while True:
        x=next(g)#获取下一个奇数
        g=filter(my_filter(x),g)
        yield x

# 函数只有被调用才会执行
for n in tt():  # 代码从这执行
    if n <100:
       print(n)
    else:
        break

