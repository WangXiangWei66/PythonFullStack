#requests：模拟浏览器的服务器服务
import requests
#在网页爬虫中，可利用etree将获取到的 HTML 页面内容解析为树形结构，方便提取所需信息。
from lxml import etree
url ='https://www.qidian.com/rank/yuepiao'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}
#_Element是用于表示 XML 或 HTML 文档中元素的基础类型
# 它包含了元素的标签、属性以及子元素等相关信息
# 可用于解析、修改和生成 XML 或 HTML 文档结构。
resp = requests.get(url,headers)
e = etree.HTML(resp.text)
print(type(e))
names=e.xpath('//div[@class="book-mid-info"]/h4/a/text()')
authors=e.xpath('//p[@class="author"]/a[1]/text()')
print(names)
print(authors)
for name,author in zip(names,authors):
    print(name,":",author)