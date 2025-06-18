import urllib.request

import urllib.error

url = 'https://movie.douban.com/'

try:
    resp = urllib.request.urlopen(url)

#HTTPError用于处理HTTP协议相关的错误
except urllib.error.HTTPError as e:
    print('原因',e.reason)
    print('相应状态码',str(e.code))
    print('响应头数据',e.headers)