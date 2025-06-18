import requests
import re

url = 'https://www.qiushibaike.com/video/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}
resp = requests.get(url, headers=headers)
# print(resp.text)
#.*表示匹配任意字符（除换行符外）的零次或多次
info = re.findall('<source src="(.*)" type = \'video/mp4\'/>', resp.text)
# print(info)
lst = []
#append:列表末尾添加一个新元素。要添加的新元素是通过字符串拼接生成的
for item in info:
    lst.append('https:' + item)
print(lst)

count = 0
for item in lst:
    count += 1
    resp = requests.get(item, headers=headers)
    #'wb'表示以二进制写入模式打开文件，这样可以处理像视频文件这种二进制数据
    with open('video/' + str(count) + '.mp4', 'wb') as file:
        file.write(resp.content)
print('视频下载完毕')
