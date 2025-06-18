import requests
from bs4 import BeautifulSoup
#案例：爬取淘宝的网页内容
url = 'https://www.taobao.com/'
headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}

resp = requests.get(url,headers)
# print(resp.text)
bs = BeautifulSoup(resp.text,'lxml')
a_list = bs.find_all('a')
print(len(a_list))
for a in a_list:
    url=a.get('href')
    # print(url)
    if url==None:
        continue
    if url.startswith('http') or url.startswith('https'):
        print(url)