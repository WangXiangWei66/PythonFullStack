import requests
from pyquery import PyQuery

url = 'https://www.qidian.com/rank/yuepiao/'
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'}
resp = requests.get(url, headers=headers)
print(resp.text)
print('---------------------')
doc = PyQuery(resp.text)
# a_tag = doc('script')
# print(a_tag)
# 列表推导式
names = [a.text for a in doc('h4 a')]
authors = doc('p.author a')

author_lst = []

for index in range(len(authors)):
    if index % 2 == 0:
        author_lst.append(authors[index].text)
print(author_lst)
print(names)
for name, author in zip(names, author_lst):
    print(name, ":", author)
