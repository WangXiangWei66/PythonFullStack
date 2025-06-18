from bs4 import BeautifulSoup

html = '''
    <html>
        <head>
            <title>马士兵教育</title>
        </head>
        <body>
            <h1 class="i bg"float='left'>欢迎来到马士兵教育</h1>
            <a href="http://www.mashibing.com">马士兵教育</a>
            <h2><!--注释的内容--></h2>
        </body>
    </html>

'''
# bs = BeautifulSoup(html, 'html.parser')
bs = BeautifulSoup(html,'lxml')
print(bs.title,type(bs.title))#获取标签
print(bs.a.attrs)#获取标签的所有属性

#获取单个属性
print(bs.h1.get('class'))
print(bs.h1['class'])
print(bs.a['href'])

#获取文本内容
print(bs.title.text)#该方法无法获取注释的文本内容
print(bs.title.string)#可以获取注释的文本内容

print('-----------',bs.h2.string)
print(bs.h2.text)

