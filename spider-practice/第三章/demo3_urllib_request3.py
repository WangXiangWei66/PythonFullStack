import urllib.request

url = 'https://movie.douban.com/'

#header:用于向服务器发送额外的信息
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}
#构建请求对象
res = urllib.request.Request(url, headers=headers)

#使用urlopen打开请求
resp = urllib.request.urlopen(res)
html = resp.read().decode('utf-8')
print(html)