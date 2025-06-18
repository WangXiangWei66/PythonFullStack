import urllib.request

url = 'https://www.baidu.com'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}
#构建请求对象
req = urllib.request.Request(url,headers = headers)

#获取opener对象
#build_opener : 用于构建自定义的OpenerDirector对象
opener = urllib.request.build_opener()
#发送请求
resp = opener.open(req)

print(resp.read().decode('utf-8'))