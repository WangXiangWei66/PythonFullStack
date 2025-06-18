import requests

url = 'https://www.so.com/s'
#params:参数（存储查询参数）
params = {'q': 'python'}
resp = requests.get(url, params=params)
resp.encoding = 'utf-8'
print(resp.text)
