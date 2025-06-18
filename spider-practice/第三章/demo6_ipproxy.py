from urllib.request import build_opener

from urllib.request import ProxyHandler
#ProxyHandler用于处理代理的操作
proxy = ProxyHandler({'http':'58.19.55.7:20082'})
#build_opener : 构建处理多种请求的对象
opener = build_opener(proxy)

url='https://www.douban.com'
resp=opener.open(url)
print(resp.read().decode('gbk'))
