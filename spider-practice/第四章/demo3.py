from bs4 import BeautifulSoup
#<span>一般用于在不产生新的换行的情况下对文本进行样式设置等操作。
html='''
    <title>马士兵教育</title>
    <div class="info"float="left">欢迎来到马士兵教育</div>
    <div class="info"float="right" id="gb">
        <span>好好学习，天天向上</span>
        <a href="http://www.mashibing.com">官网</a>
    </div>
    <span>人生苦短，你需要Python</span>
'''

bs = BeautifulSoup(html,'lxml')
# print(bs.title, type(bs.title))
print(bs.find('div', class_='info'),"+",type(bs.find('div',class_='info')))
print("--------------")
#find_all方法得到的是一个标签的列表
print(bs.find_all('div', class_='info'))
print("--------------")
for item in bs.find_all('div',class_='info'):
    print(item,"+",type(item))
print("--------------")
print(bs.find_all('div',attrs={'float':'right'}))

print("------CSS选择器--------")
print(bs.select("#gb"))#通过ID查找
print("--------------")
print(bs.select('.info'))#通过class属性查找
print("--------------")
print(bs.select('div>span'))#打印div下的span属性
print("--------------")
print(bs.select('div.info>span'))

print("--------------")
for item in bs.select('div.info>span'):
    print(item.text)


