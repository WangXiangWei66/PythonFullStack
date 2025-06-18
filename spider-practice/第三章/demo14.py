import requests

url = ('https://fanqienovel.com/')
#login表示请求意图是登录
data = {'username': '15554007467', 'password': '2293767663', 'action': 'login'}
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}
resp = requests.post(url, data=data,headers=headers)
resp.encoding = 'gb2311'
print(resp.status_code)
print(resp.text)
