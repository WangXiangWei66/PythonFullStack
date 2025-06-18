#pyquery是一个用于解析和操作 HTML、XML 文档的库
from pyquery import PyQuery as py
html= '''
    <html>
        <head>
            <title>Pyquery</title>
        </head>
        <body>
            <h1>Pyquery</h1>
        </body>
    </html>
'''
#创建pyQuery对象，实际上就是将str类型转化为pyQuery类型
doc = py(html)
print(doc)
print('-------------')
print(type(doc))
print('-------------')
print(type(html))
print('-------------')
print(doc('title'))
print('-------本demo是演示的字符串方式-------')