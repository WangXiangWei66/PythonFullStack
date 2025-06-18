#request用于处理HTTP请求
import requests

url = 'http://www.baidu.com'

resp = requests.get(url)

resp.encoding = 'utf-8'
cookie = resp.cookies
headers = resp.headers

print('相应状态码',resp.status_code)
print('请求后的cookie:',cookie)
print('获取请求的网址',resp.url)
print('响应头',headers)
print('相应内容',resp.text)