from pyquery import PyQuery
#这个演示的是URL方式
doc = PyQuery(url='http://www.baidu.com', encoding='utf-8')
print(doc('title'))
