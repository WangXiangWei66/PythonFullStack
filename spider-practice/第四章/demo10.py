
from pyquery import PyQuery
html='''
    <html>
        <head>
            <title>PyQuery</title>
        </head>
        <body>
            <div id  = "main">
                <a href = "http://www.mashibing.com">马士兵教育</a>
                <h1>欢迎来到马士兵教育</h1>
                我是div中的文本
            </div>
            <h2>Python学习</h2>
        </body>
    </html>
'''
doc = PyQuery(html)
print(doc("#main"))#获取当前节点

print('----------父节点---------')
#打印的是整个Body
print(doc("#main").parent())
print("--------子节点---------")
print(doc("#main").children())
print('---------兄弟节点---------')
print(doc("#main").siblings())

print('---------获取属性-----------')
print(doc("a").attr('href'))
print('-----------获取标签的内容----------')
print(doc('#main').html())
print('-----------')
print(doc('#main').text())