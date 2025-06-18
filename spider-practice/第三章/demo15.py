import requests

url = 'https://www.xslou.com/login.php'
data = {'username': '18600605736', 'password': '57365736', 'action': 'login'}

#创建session对象并复制给session变量
#session可以跨请求保存一些参数
session = requests.session()
resp = session.post(url, data=data)
#gb2312是汉字编码标准，是简体中文编码
resp.encoding = 'gb2312'

hot_url = 'https://www.xslou.com/modules/article/uservote.php?id=71960'
resp2 = session.get(hot_url)
resp2.encoding = 'gb2312'
print(resp2.text)
